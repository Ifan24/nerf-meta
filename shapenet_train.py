import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from models.rendering import get_rays_shapenet, sample_points, volume_render
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
from torchmeta.utils.gradient_based import gradient_update_parameters as GUP
import matplotlib.pyplot as plt


def inner_loop(model, optim, imgs, poses, hwf, bound, num_samples, raybatch_size, inner_steps):
    """
    train the inner model for a specified number of iterations
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in range(inner_steps):
        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()
    
def report_result(model, imgs, poses, hwf, bound, num_samples, raybatch_size, tto_showImages):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)
    view_psnrs = []
    plt.figure(figsize=(15, 6))
    count = 0
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, raybatch_size):
                rgbs_batch, sigmas_batch = model(xyz[i:i+raybatch_size])
                color_batch = volume_render(rgbs_batch, sigmas_batch, 
                                            t_vals[i:i+raybatch_size],
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
                plt.title(f'synth psnr:{psnr:0.2f}')
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
    parser.add_argument('--meta', type=str, default='Reptile', choices=['MAML', 'Reptile'],
                        help='meta algorithm, (MAML, Reptile)')
    parser.add_argument('--MAML_batch', type=int, default=3,
                        help='number of batch of task for MAML')
    args = parser.parse_args()
    
    use_reptile = args.meta == 'Reptile'
    
    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load train & val dataset
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

    meta_model = build_nerf(args)
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
    train_psnrs = []
    while step < args.max_iters:
        for imgs, poses, hwf, bound in train_loader:
            # imgs = [1, train_views(25), H(128), W(128), C(3)]
            imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
            imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()
    
            meta_optim.zero_grad()
            
            if use_reptile:
                inner_model = copy.deepcopy(meta_model)
                inner_optim = torch.optim.SGD(inner_model.parameters(), args.inner_lr)
        
                inner_loop(inner_model, inner_optim, imgs, poses,
                            hwf, bound, args.num_samples,
                            args.train_batchsize, args.inner_steps)
                            
                # Reptile
                with torch.no_grad():
                    for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                        meta_param.grad = meta_param - inner_param
            # python shapenet_train.py --config configs/shapenet/chairs.json --meta MAML
            # MAML
            # https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/train.py
            else:
                def MAML_inner_loop(model, pixels, rays_o, rays_d, num_rays, bound, num_samples, raybatch_size, inner_steps, alpha=5e-2):
                    params = None
                
                    # Multi-Step Loss Optimization
                    total_loss = torch.tensor(0.).to(device)
                    for step in range(inner_steps+1):
                        indices = torch.randint(num_rays, size=[raybatch_size])
                        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
                        pixelbatch = pixels[indices] 
                        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                                    num_samples, perturb=True)
                        
                        rgbs, sigmas = model(xyz, params=params)
                        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
                        loss = F.mse_loss(colors, pixelbatch)
                        
                        total_loss += loss * step/inner_steps
                        
                        if step == inner_steps:
                            return total_loss
                        
                        model.zero_grad()
                        params = GUP(model, loss, params=params, step_size=alpha, first_order=True)
                        
                            
                outer_loss = torch.tensor(0.).to(device)
                batch_size = args.MAML_batch
                pixels = imgs.reshape(-1, 3)
                # 25x128x128
                rays_o, rays_d = get_rays_shapenet(hwf, poses)
                rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
                num_rays = rays_d.shape[0]
                
                # In MAML, the losses of a batch tasks were used to update meta parameter 
                # but the batch of tasks in NeRF does not makes too much sense
                # should it be a batch of scenes? or a batch of pixels in a single scene
                for i in range(batch_size):
                    # update parameter with the inner loop loss
                    loss = MAML_inner_loop(meta_model, pixels, rays_o, rays_d, num_rays, bound, args.num_samples,
                            args.train_batchsize, args.inner_steps, args.inner_lr)
                    
                    outer_loss += loss
                    
                meta_optim.zero_grad()
                outer_loss.div_(batch_size*args.inner_steps)
                outer_loss.backward()
        
            meta_optim.step()
        
            if step % args.val_freq == 0 and step != args.resume_step:
                train_psnrs.append((step, -10*torch.log10(outer_loss).detach().cpu()))
                
                val_psnr = val_meta(args, meta_model, val_loader, device)
                print(f"step: {step}, val psnr: {val_psnr:0.3f}")
                val_psnrs.append((step, val_psnr.cpu()))
                
                plt.subplots()
                plt.plot(*zip(*val_psnrs), label="Meta learning validation PSNR")
                plt.plot(*zip(*train_psnrs), label="Meta learning Training PSNR")
                plt.title('ShapeNet Meta learning Training PSNR')
                plt.xlabel('Iterations')
                plt.ylabel('PSNR')
                plt.legend()
                plt.show()
          
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