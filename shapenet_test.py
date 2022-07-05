from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf, build_MipNerf
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render, get_d, get_rays_shapenet_mipNerf, get_raybatch
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
from shapenet_train import report_result, inner_loop, calc_psnr, render_image
from models.rendering import Rays_keys, Rays
from livelossplot import PlotLosses

def test_time_optimize(args, model, optim, imgs, poses, hwf, bound):
    """
    test-time-optimize the meta trained model on available views
    """
    inner_loop(model, optim, imgs, poses, hwf, bound, args.num_samples, args.tto_batchsize, args.tto_steps)

def train_val_scene(args, model, optim, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound):
    """
    train and val the model on available views
    """
    coarse_loss_mult = 0.2
    rays = get_rays_shapenet_mipNerf(hwf, tto_poses)
    raybatch, val_mask = get_raybatch(rays, tto_imgs, args.tto_batchsize)
    indices = torch.randint(len(raybatch), size=[args.train_val_steps])
    
    test_rays = get_rays_shapenet_mipNerf(hwf, test_poses)
    train_val_freq = args.train_val_freq
    
    # plt_groups = {'Test PSNR':[]}
    # plotlosses_model = PlotLosses(groups=plt_groups)
    # plt_groups['Test PSNR'].append('test')
    val_psnrs = []
    for step in tqdm(range(args.train_val_steps), desc = 'Train & Validate'):
        idx = indices[step]
        rgbs = raybatch[idx].color
        rays_batch = raybatch[idx]
        
        optim.zero_grad()
        
        ret = model(rays_batch, True)
        # calculate loss for coarse and fine
        mask = rays_batch.lossmult
        losses = []
        for (rgb, _, _) in ret:
            losses.append((mask * (rgb - rgbs[..., :3]) ** 2).sum() / mask.sum())
        # The loss is a sum of coarse and fine MSEs
        mse_corse, mse_fine = losses
        loss = coarse_loss_mult * mse_corse + mse_fine
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            # validation
            if step % train_val_freq == 0 and step != 0:
                view_psnrs = []
                plt.figure(figsize=(15, 6))
                count = 0
                for i in range(test_imgs.shape[0]):
                    img = test_imgs[i]
                    ray = Rays(
                        origins=test_rays.origins[i],
                        directions=test_rays.directions[i],
                        viewdirs=test_rays.viewdirs[i],
                        radii=test_rays.radii[i],
                        lossmult=test_rays.lossmult[i],
                        near=test_rays.near[i],
                        far=test_rays.far[i]
                    )
                    
                    rgb_gt = img[..., :3]
                    coarse_rgb, fine_rgb, val_mask = render_image(model, ray, img, args.test_batchsize)
                
                    val_mse_coarse = (val_mask * (coarse_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
                    val_mse_fine = (val_mask * (fine_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
                
                    val_loss = coarse_loss_mult * val_mse_coarse + val_mse_fine
                    val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)
                    view_psnrs.append(val_psnr_fine)
                
                    fine_rgb = torch.clip(fine_rgb.squeeze(0), min=0, max=1)
                
                    if count < args.tto_showImages:
                        plt.subplot(2, 5, count+1)
                        plt.imshow(img.cpu())
                        plt.title('Target')
                        plt.subplot(2,5,count+6)
                        plt.imshow(fine_rgb.cpu())
                        plt.title(f'{val_psnr_fine:0.2f}')
                    count += 1
                    
                plt.show()
                
                plt.figure()
                scene_psnr = torch.stack(view_psnrs).mean().item()
                val_psnrs.append((step, scene_psnr))
                plt.plot(*zip(*val_psnrs), label="val_psnr")
                plt.title('ShapeNet Reconstruction from 25 views')
                plt.xlabel('Iterations')
                plt.ylabel('PSNR')
                plt.legend()
                plt.show()

                # plotlosses_model.update({'test':scene_psnr}, current_step=step)
                # plotlosses_model.send()
            
            # step           = 100 -> 1000 -> 10000 -> 50000
            # train_val_freq = 100 -> 500  -> 2500  -> 5000
            if step <= 1000:
                train_val_freq = 100
            elif step > 1000 and step <= 10000:
                train_val_freq = 500
            elif step > 10000 and step <= 50000:
                train_val_freq = 2500
            elif step > 50000 and step <= 100000:
                train_val_freq = 5000
                
    print(val_psnrs)
    
def test():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                    help='config file for the shape class (cars, chairs or lamps)')    
    parser.add_argument('--weight-path', type=str, required=True,
                        help='path to the meta-trained weight file')
    parser.add_argument('--one_scene', action='store_true', help="train and validate the model on the first scene of test dataset")
    parser.add_argument('--standard_init', action='store_true', help="train and validate the model without meta learning parameters")
    
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = build_shapenet(image_set="test", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    test_set = Subset(test_set, range(0, args.max_test_size))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_MipNerf()
    model.to(device)

    if not args.standard_init:
        checkpoint = torch.load(args.weight_path, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']

    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True)
    
    if args.one_scene:
        for imgs, poses, hwf, bound in test_loader:
            imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
            imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
            tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
            tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)
            
            if not args.standard_init:
                model.load_state_dict(meta_state)
                
            optim = torch.optim.SGD(model.parameters(), args.tto_lr)
            train_val_scene(args, model, optim, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound)
            return
    
    test_psnrs = []
    idx = 0
    pbar = tqdm(test_loader, desc = 'Testing')
    for imgs, poses, hwf, bound in pbar:
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        model.load_state_dict(meta_state)
        optim = torch.optim.SGD(model.parameters(), args.tto_lr)

        test_time_optimize(args, model, optim, tto_imgs, tto_poses, hwf, bound)
        # scene_psnr = report_result(args, model, test_imgs, test_poses, hwf, bound)
        scene_psnr = report_result(model, test_imgs, test_poses, hwf, bound, 
                                    args.num_samples, args.test_batchsize, args.tto_showImages)
        
        if args.create_video:
            create_360_video(args, model, hwf, bound, device, idx+1, savedir)
            print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        else:
            print(f"scene {idx+1}, psnr:{scene_psnr:.3f}")
            
        test_psnrs.append(scene_psnr)
        
        pbar.set_postfix(mean_psnr=torch.stack(test_psnrs).mean().item())
        idx += 1
    
    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")


if __name__ == '__main__':
    test()