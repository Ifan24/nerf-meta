import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf, build_MipNerf
from models.rendering import get_rays_shapenet, sample_points, volume_render, get_d, get_rays_shapenet_mipNerf, get_raybatch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt

def calc_mse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    return torch.mean((x - y) ** 2)
    
def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr

def inner_loop(model, optim, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    # mult factor for coarse MLP loss
    coarse_loss_mult = 0.2
    # imgs = [N, H, W, 3]
    
    rays = get_rays_shapenet_mipNerf(hwf, poses)
    # rays = [N, H, W, 3 or 1]
    raybatch, val_mask = get_raybatch(rays, imgs, raybatch_size)
    # len(raybatch) = N*H*W/raybatch_size
    # raybatch.origins = [raybatch_size, 3]
    # randomly select N=inner_steps sample from raybatch
    indices = torch.randint(len(raybatch), size=[inner_steps])
    
    for step in range(inner_steps):
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
        
        # with torch.no_grad():
        #     psnrs = []
        #     for (rgb, _, _) in ret:
        #         psnrs.append(calc_psnr(rgb, rgbs[..., :3]))
        #     psnr_corse, psnr_fine = psnrs
            
            
    # rays_o, rays_d = get_rays_shapenet(hwf, poses)
    # rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    # for step in range(inner_steps):
    #     indices = torch.randint(num_rays, size=[raybatch_size])
    #     raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
    #     pixelbatch = pixels[indices] 
    #     t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
    #                                 num_samples, perturb=True)
    #     # xyz = [B, N, 3]
    #     input_shape = xyz.shape
    #     viewdirs = get_d(raybatch_d, xyz.shape[-2])
        
    #     optim.zero_grad()
    #     rgbs, sigmas = model(xyz, viewdirs)
        
    #     colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        # loss = F.mse_loss(colors, pixelbatch)
        # loss.backward()
        # optim.step()

def report_result(model, imgs, poses, hwf, bound, num_samples, raybatch_size, show_one):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    num_samples, perturb=False)
        viewdirs = get_d(rays_d, xyz.shape[-2])
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, raybatch_size):
                input_shape = xyz[i:i+raybatch_size].shape
                xyz_batch = xyz[i:i+raybatch_size]
                xyz_batch = xyz_batch.reshape([-1,3])

                viewdir_batch = viewdirs[i:i+raybatch_size]
                viewdir_batch = viewdir_batch.reshape([-1,3])

                sigmas, rgbs = model(xyz_batch, viewdir_batch)

                # unflatten batch ray
                sigmas_batch = sigmas.reshape(input_shape[:-1])
                rgbs_batch = rgbs.reshape(input_shape)

                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+raybatch_size],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.clip(torch.cat(synth, dim=0).reshape_as(img), min=0, max=1)
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
    
    scene_psnr = torch.stack(view_psnrs).mean()
    
    if show_one:
      plt.figure(figsize=(15, 5))   
      plt.subplot(1, 2, 1)
      plt.imshow(img.cpu())
      plt.subplot(1, 2, 2)
      plt.imshow(synth.cpu())
      plt.title(f'psnr:{scene_psnr:0.3f}')
      plt.show()
      
    return scene_psnr

def val_meta(args, model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    meta_trained_state = model.state_dict()
    val_model = copy.deepcopy(model)
    # show one of the validation result
    show_one = True
    val_psnrs = []
    for imgs, poses, hwf, bound in tqdm(val_loader, desc = 'Validating'):
        imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
        imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

        tto_imgs, test_imgs = torch.split(imgs, [args.tto_views, args.test_views], dim=0)
        tto_poses, test_poses = torch.split(poses, [args.tto_views, args.test_views], dim=0)

        val_model.load_state_dict(meta_trained_state)
        val_optim = torch.optim.SGD(val_model.parameters(), args.tto_lr)

        inner_loop(val_model, val_optim, tto_imgs, tto_poses, hwf,
                    bound, args.num_samples, args.tto_batchsize, args.tto_steps)
        
        scene_psnr = report_result(val_model, test_imgs, test_poses, hwf, bound, 
                                    args.num_samples, args.test_batchsize, show_one)
        show_one = False
        val_psnrs.append(scene_psnr)

    val_psnr = torch.stack(val_psnrs).mean()
    return val_psnr


def main():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the shape class (cars, chairs or lamps)')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_shapenet(image_set="train", dataset_root=args.dataset_root,
                                splits_path=args.splits_path, num_views=args.train_views)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenet(image_set="val", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # meta_model = build_nerf(args)
    meta_model = build_MipNerf()
    meta_model.to(device)

    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    
    step = 0
    pbar = tqdm(total=args.max_iters, desc = 'Training')
    val_psnrs = []
    
    while step < args.max_iters:
        for imgs, poses, hwf, bound in train_loader:
            # imgs = [1, train_views(25), H(128), W(128), C(3)]
            imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
            imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
    
            meta_optim.zero_grad()
    
            inner_model = copy.deepcopy(meta_model)
            inner_optim = torch.optim.SGD(inner_model.parameters(), args.inner_lr)
    
            inner_loop(inner_model, inner_optim, imgs, poses,
                        hwf, bound, args.num_samples,
                        args.train_batchsize, args.inner_steps)
            
            with torch.no_grad():
                for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                    meta_param.grad = meta_param - inner_param
            
            meta_optim.step()
        
        if step % args.val_freq == 0 and step != 0:
            val_psnr = val_meta(args, meta_model, val_loader, device)
            print(f"step: {step}, val psnr: {val_psnr:0.3f}")
            val_psnrs.append(val_psnr)
      
        if step % args.checkpoint_freq == 0 and step != 0:
            path = f"{args.checkpoint_path}/step{step}.pth"
            torch.save({
                'epoch': step,
                'meta_model_state_dict': meta_model.state_dict(),
                'meta_optim_state_dict': meta_optim.state_dict(),
                }, path)
        
        step += 1
        pbar.update(1)
        
        if step > args.max_iters:
          break
        
    pbar.close()
    

if __name__ == '__main__':
    main()