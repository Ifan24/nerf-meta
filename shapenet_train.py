import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf, build_MipNerf
from models.rendering import get_rays_shapenet, sample_points, volume_render, get_d, get_rays_shapenet_mipNerf, get_raybatch
from models.mip import rearrange_render_image
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
from models.rendering import Rays_keys, Rays


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
    # return psnr_fine
            

def render_image(model, rays, rgbs, val_chunk_size):
    height, width, _ = rgbs.shape  # H W C
    single_image_rays, val_mask = rearrange_render_image(rays, val_chunk_size)
    coarse_rgb, fine_rgb = [], []
    with torch.no_grad():
        for batch_rays in single_image_rays:
            (c_rgb, _, _), (f_rgb, _, _) = model(batch_rays, False)
            coarse_rgb.append(c_rgb)
            fine_rgb.append(f_rgb)

    coarse_rgb = torch.cat(coarse_rgb, dim=0)
    fine_rgb = torch.cat(fine_rgb, dim=0)

    coarse_rgb = coarse_rgb.reshape(1, height, width, coarse_rgb.shape[-1])  # N H W C
    fine_rgb = fine_rgb.reshape(1, height, width, fine_rgb.shape[-1])  # N H W C
    return coarse_rgb, fine_rgb, val_mask
        
    
def report_result(model, imgs, poses, hwf, bound, num_samples, raybatch_size, tto_showImages):
    """
    report view-synthesis result on heldout views
    """
    coarse_loss_mult = 0.2
    rays = get_rays_shapenet_mipNerf(hwf, poses)
    
    view_psnrs = []
    fig = plt.figure(figsize=(15, 6))
    count = 0
    for i in range(imgs.shape[0]):
        img = imgs[i]
        ray = Rays(
            origins=rays.origins[i],
            directions=rays.directions[i],
            viewdirs=rays.viewdirs[i],
            radii=rays.radii[i],
            lossmult=rays.lossmult[i],
            near=rays.near[i],
            far=rays.far[i]
        )
        
        rgb_gt = img[..., :3]
        coarse_rgb, fine_rgb, val_mask = render_image(model, ray, img, raybatch_size)
    
        val_mse_coarse = (val_mask * (coarse_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
        val_mse_fine = (val_mask * (fine_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
    
        val_loss = coarse_loss_mult * val_mse_coarse + val_mse_fine
        val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)
        view_psnrs.append(val_psnr_fine)
    
        fine_rgb = torch.clip(fine_rgb.squeeze(0), min=0, max=1)
        
        if count < tto_showImages:
            plt.subplot(2, 5, count+1)
            plt.imshow(img.cpu())
            plt.title('Target')
            plt.subplot(2,5,count+6)
            plt.imshow(fine_rgb.cpu())
            plt.title(f'synth psnr:{val_psnr_fine:0.2f}')
        count += 1
        
    plt.show()
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr

def val_meta(args, model, val_loader, device):
    """
    validate the meta trained model for few-shot view synthesis
    """
    meta_trained_state = model.state_dict()
    val_model = copy.deepcopy(model)
    # show one of the validation result
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
                                    args.num_samples, args.test_batchsize, args.tto_showImages)
        val_psnrs.append(scene_psnr)

    val_psnr = torch.stack(val_psnrs).mean()
    return val_psnr


def main():
    parser = argparse.ArgumentParser(description='shapenet few-shot view synthesis')
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the shape class (cars, chairs or lamps)')
    parser.add_argument('--resume_step', type=int, default=0,
                        help='resume training from step')
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_shapenet(image_set="train", dataset_root=args.dataset_root,
                                splits_path=args.splits_path, num_views=args.train_views)
    if args.max_train_size != 0:
        train_set = Subset(train_set, range(0, args.max_train_size))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    val_set = build_shapenet(image_set="val", dataset_root=args.dataset_root,
                            splits_path=args.splits_path,
                            num_views=args.tto_views+args.test_views)
    if args.max_val_size != 0:
        val_set = Subset(val_set, range(0, args.max_val_size))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # meta_model = build_nerf(args)
    meta_model = build_MipNerf()
    meta_model.to(device)
    
    if args.resume_step != 0:
        weight_path = f"{args.checkpoint_path}/step{args.resume_step}.pth"
        checkpoint = torch.load(weight_path, map_location=device)
        meta_state = checkpoint['meta_model_state_dict']
        meta_model.load_state_dict(meta_state)


    meta_optim = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    
    step = args.resume_step
    pbar = tqdm(total=args.max_iters, desc = 'Training')
    pbar.update(args.resume_step)
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
        
            if step % args.val_freq == 0 and step != args.resume_step:
                val_psnr = val_meta(args, meta_model, val_loader, device)
                print(f"step: {step}, val psnr: {val_psnr:0.3f}")
                val_psnrs.append((step, val_psnr))
          
            if step % args.checkpoint_freq == 0 and step != args.resume_step:
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
    print(val_psnrs)
    

if __name__ == '__main__':
    main()