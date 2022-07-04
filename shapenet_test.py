from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render, get_d
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt


def test_time_optimize(args, model, optim, imgs, poses, hwf, bound):
    """
    test-time-optimize the meta trained model on available views
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    tepoch = tqdm(range(args.tto_steps), desc = 'Training')
    for step in tepoch:
        indices = torch.randint(num_rays, size=[args.tto_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        input_shape = xyz.shape
        viewdirs = get_d(raybatch_d, xyz.shape[-2])
        
        optim.zero_grad()
        
        # flatten batch ray
        xyzs = xyzs.reshape([-1,3])
        viewdirs = viewdirs.reshape([-1,3])
        
        sigmas, rgbs = model(xyzs, viewdirs)
        
        # unflatten batch ray
        sigmas = sigmas.reshape(input_shape[:-1])
        rgbs = rgbs.reshape(input_shape)
        
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        tepoch.set_postfix(loss=loss.item())
        loss.backward()
        optim.step()


def report_result(args, model, imgs, poses, hwf, bound):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    
    if args.tto_views == 1:
        plt.imshow(imgs[0])
        plt.title('Input Image')
        plt.show()
        
    plt.figure(figsize=(15,6))
    count = 0
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    args.num_samples, perturb=False)
        viewdirs = get_d(rays_d, xyz.shape[-2])
        synth = []
        num_rays = rays_d.shape[0]
        
        with torch.no_grad():
            for i in range(0, num_rays, args.test_batchsize):
                input_shape = xyz[i:i+args.test_batchsize].shape
                xyz_batch = xyz[i:i+args.test_batchsize]
                xyz_batch = xyz_batch.reshape([-1,3])

                viewdir_batch = viewdirs[i:i+args.test_batchsize]
                viewdir_batch = viewdir_batch.reshape([-1,3])

                sigmas, rgbs = model(xyz_batch, viewdir_batch)

                # unflatten batch ray
                sigmas_batch = sigmas.reshape(input_shape[:-1])
                rgbs_batch = rgbs.reshape(input_shape)

                color_batch = volume_render(rgbs_batch, sigmas_batch,
                                            t_vals[i:i+args.test_batchsize],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.clip(torch.cat(synth, dim=0).reshape_as(img), min=0, max=1)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
            
        if count < args.tto_showImages:
            plt.subplot(2,5,count+1)
            plt.imshow(img.cpu())
            plt.title('Target')
            plt.subplot(2,5,count+6)
            plt.imshow(synth.cpu())
            plt.title('Reconstruction')
        count += 1
    plt.show()
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def test():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                    help='config file for the shape class (cars, chairs or lamps)')    
    parser.add_argument('--weight-path', type=str, required=True,
                        help='path to the meta-trained weight file')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = build_shapenet(image_set="test", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = build_nerf(args)
    model.to(device)

    checkpoint = torch.load(args.weight_path, map_location=device)
    meta_state = checkpoint['meta_model_state_dict']

    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True)
    
    test_psnrs = []
    for idx, (imgs, poses, hwf, bound) in enumerate(test_loader):
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        model.load_state_dict(meta_state)
        optim = torch.optim.SGD(model.parameters(), args.tto_lr)

        test_time_optimize(args, model, optim, tto_imgs, tto_poses, hwf, bound)
        scene_psnr = report_result(args, model, test_imgs, test_poses, hwf, bound)

        create_360_video(args, model, hwf, bound, device, idx+1, savedir)
        
        print(f"scene {idx+1}, psnr:{scene_psnr:.3f}, video created")
        test_psnrs.append(scene_psnr)
    
    test_psnrs = torch.stack(test_psnrs)
    print("----------------------------------")
    print(f"test dataset mean psnr: {test_psnrs.mean():.3f}")


if __name__ == '__main__':
    test()