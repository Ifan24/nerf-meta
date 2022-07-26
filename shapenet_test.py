from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
from shapenet_train import report_result, inner_loop

def test_time_optimize(args, model, optim, imgs, poses, hwf, bound):
    """
    test-time-optimize the meta trained model on available views
    """
    inner_loop(model, optim, imgs, poses, hwf, bound, args.num_samples, args.tto_batchsize, args.tto_steps)


def train_val_scene(args, model, optim, tto_imgs, tto_poses, test_imgs, test_poses, hwf, bound):
    """
    train and val the model on available views
    """
    train_val_freq = args.train_val_freq
    pixels = tto_imgs.reshape(-1, 3)

    tto_rays_o, tto_rays_d = get_rays_shapenet(hwf, tto_poses)
    tto_rays_o, tto_rays_d = tto_rays_o.reshape(-1, 3), tto_rays_d.reshape(-1, 3)

    test_ray_origins, test_ray_directions = get_rays_shapenet(hwf, test_poses)

    num_samples = args.num_samples
    tto_showImages = 5    

    num_rays = tto_rays_d.shape[0]
    val_psnrs = []
    for step in tqdm(range(args.train_val_steps), desc = 'Train & Validate'):
        indices = torch.randint(num_rays, size=[args.tto_batchsize])
        raybatch_o, raybatch_d = tto_rays_o[indices], tto_rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()

        if step % train_val_freq == 0 and step != 0:
            with torch.no_grad():
                view_psnrs = []
                plt.figure(figsize=(15, 6))
                count = 0
                
                for img, rays_o, rays_d in zip(test_imgs, test_ray_origins, test_ray_directions):
                    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
                    t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                                num_samples, perturb=False)
                    
                    synth = []
                    test_num_rays = rays_d.shape[0]
                    for i in range(0, test_num_rays, args.test_batchsize):
                        rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize])
                        color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                                    t_vals[i:i+args.test_batchsize],
                                                    white_bkgd=True)
                        synth.append(color_batch)
                    
                    synth = torch.clip(torch.cat(synth, dim=0).reshape_as(img), min=0, max=1)
                    error = F.mse_loss(img, synth)
                    psnr = -10*torch.log10(error)
                    view_psnrs.append(psnr)
                    
                    if count < tto_showImages:
                        plt.subplot(2, 5, count+1)
                        plt.imshow(img.cpu())
                        plt.title('Target')
                        plt.subplot(2,5,count+6)
                        plt.imshow(synth.cpu())
                        plt.title(f'{psnr:0.2f}')   
                    count += 1

                plt.show()
              
                plt.figure()
                scene_psnr = torch.stack(view_psnrs).mean().item()
                val_psnrs.append((step, scene_psnr))
                print(f"step: {step}, val psnr: {scene_psnr:0.3f}")
                plt.plot(*zip(*val_psnrs), label="val_psnr")
                plt.title(f'ShapeNet Reconstruction from {args.tto_views} views')
                plt.xlabel('Iterations')
                plt.ylabel('PSNR')
                plt.legend()
                plt.show()

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
    if args.max_test_size != 0:
        test_set = Subset(test_set, range(0, args.max_test_size))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_nerf(args)
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