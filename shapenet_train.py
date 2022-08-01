import argparse
import json
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from models.rendering import get_rays_shapenet, sample_points, volume_render
try:
  import google.colab
  from tqdm.notebook import tqdm as tqdm
except:
  from tqdm import tqdm as tqdm
  
import matplotlib.pyplot as plt
from torchmeta.utils.gradient_based import gradient_update_parameters as GUP
import matplotlib.pyplot as plt
from collections import OrderedDict

def prepare_MAML_data(imgs, poses, batch_size, hwf):
    '''
        split training images to support set and target set
        size(support set) = train_views - MAML_batch_size
        size(target set) = MAML_batch_size
    '''
    
    target_imgs, target_poses = imgs[:batch_size], poses[:batch_size]
    imgs, poses = imgs[batch_size:], poses[batch_size:] 
    
    pixels = imgs.reshape(-1, 3)
    # 25x128x128
    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    num_rays = rays_d.shape[0]
    
    target_pixels = target_imgs.reshape(-1, 3)
    target_rays_o, target_rays_d = get_rays_shapenet(hwf, target_poses)
    target_rays_o, target_rays_d = target_rays_o.reshape(-1, 3), target_rays_d.reshape(-1, 3)
    target_num_rays = target_rays_d.shape[0]
    
    return {
            'support': [pixels, rays_o, rays_d, num_rays], 
            'target':[target_pixels, target_rays_o, target_rays_d, target_num_rays]
        }
                    
def MAML_inner_loop(model, bound, num_samples, raybatch_size, inner_steps, 
    alpha, train_data):
    pixels, rays_o, rays_d, num_rays = train_data['support']
    target_pixels, target_rays_o, target_rays_d, target_num_rays = train_data['target']
    
    params = None

    for step in range(inner_steps):
        indices = torch.randint(num_rays, size=[raybatch_size])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    num_samples, perturb=True)
        
        rgbs, sigmas = model(xyz, params=params)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)

        model.zero_grad()
        params = GUP(model, loss, params=params, step_size=alpha)
        
    # use the param from previous inner_steps on val views to get a outer loss
    indices = torch.randint(target_num_rays, size=[raybatch_size])
    raybatch_o, raybatch_d = target_rays_o[indices], target_rays_d[indices]
    pixelbatch = target_pixels[indices] 
    t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                num_samples, perturb=True)
    
    rgbs, sigmas = model(xyz, params=params)
    colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
    loss = F.mse_loss(colors, pixelbatch)
             
    return loss
                    
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
    # Meta-SGD
    parser.add_argument('--learn_step_size', action='store_true',
                        help='the step size is a learnable (meta-trained) additional argument')
    # Meta-SGD
    parser.add_argument('--per_param_step_size', action='store_true',
                        help='the step size parameter is different for each parameter of the model. Has no impact unless `learn_step_size=True')
    parser.add_argument('--reptile_torchmeta', action='store_true',
                        help='use torchmeta framework for reptile inner loop')
    # MAML++
    parser.add_argument('--use_scheduler', action='store_true',
                        help='use scheduler to adjust outer loop lr')   
                        
    args = parser.parse_args()
    
    
    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value
            
    print(args)
    use_reptile = args.meta == 'Reptile'
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
    
    # learn_step_size & per_param_step_size
    step_size = args.inner_lr
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=meta_optim, T_max=args.max_iters/50, eta_min=args.meta_lr/100)
                                                              
    if args.per_param_step_size:
        step_size = OrderedDict((name, torch.tensor(step_size,
            dtype=param.dtype, device=device,
            requires_grad=args.learn_step_size)) for (name, param)
            in meta_model.meta_named_parameters())
    else:
        step_size = torch.tensor(step_size, dtype=torch.float32,
            device=device, requires_grad=args.learn_step_size)
            
    if args.learn_step_size:
        meta_optim.add_param_group({'params': step_size.values()
            if args.per_param_step_size else [step_size]})
    
        # outer loop lr
        # if scheduler is not None:
        #     for group in meta_optim.param_groups:
        #         group.setdefault('initial_lr', group['lr'])
        #     # scheduler.base_lrs([group['initial_lr'] for group in meta_optim.param_groups])
        #     print([group['initial_lr'] for group in meta_optim.param_groups])
            
    
    
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
                if args.reptile_torchmeta:
                    # OOM if inner step > 16
                    params = None
                    
                    pixels = imgs.reshape(-1, 3)
                    rays_o, rays_d = get_rays_shapenet(hwf, poses)
                    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
                    num_rays = rays_d.shape[0]
                    
                    for step in range(args.inner_steps):
                        indices = torch.randint(num_rays, size=[args.train_batchsize])
                        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
                        pixelbatch = pixels[indices] 
                        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                                    args.num_samples, perturb=True)
                        
                        rgbs, sigmas = meta_model(xyz, params=params)
                        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
                        loss = F.mse_loss(colors, pixelbatch)
                        meta_model.zero_grad()
                        params = GUP(meta_model, loss, params=params, step_size=step_size)
                        
                    pbar.set_postfix({
                        'inner_lr': step_size['net.1.weight'].item() if args.per_param_step_size else step_size.item(), 
                        "outer_lr" : scheduler.get_last_lr()[0], 
                        'Train loss': loss.item()
                    })
                    # Reptile
                    with torch.no_grad():
                        for meta_param, inner_param in zip(meta_model.meta_parameters(), params.items()):
                            meta_param.grad = meta_param - inner_param[1]
                else:
                    inner_model = copy.deepcopy(meta_model)
                    inner_optim = torch.optim.SGD(inner_model.parameters(), args.inner_lr)
            
                    inner_loop(inner_model, inner_optim, imgs, poses,
                                hwf, bound, args.num_samples,
                                args.train_batchsize, args.inner_steps)
                                
                    # Reptile
                    with torch.no_grad():
                        for meta_param, inner_param in zip(meta_model.parameters(), inner_model.parameters()):
                            meta_param.grad = meta_param - inner_param
            # python shapenet_train.py --config configs/shapenet/chairs.json --meta MAML --learn_step_size --per_param_step_size 
            # MAML
            # https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/train.py
            else:                            
                outer_loss = torch.tensor(0.).to(device)
                batch_size = args.MAML_batch
                train_data = prepare_MAML_data(imgs, poses, batch_size, hwf)
                                    
                # In MAML, the losses of a batch tasks were used to update meta parameter 
                # but the batch of tasks in NeRF does not makes too much sense
                # should it be a batch of scenes? or a batch of pixels in a single scene
                for i in range(batch_size):
                    # update parameter with the inner loop loss
                    loss = MAML_inner_loop(meta_model, bound, args.num_samples,
                            args.train_batchsize, args.inner_steps, step_size, train_data)
                    
                    pbar.set_postfix({
                        'inner_lr': step_size['net.1.weight'].item() if args.per_param_step_size else step_size.item(), 
                        "outer_lr" : scheduler.get_last_lr()[0], 
                        'Train loss': loss.item()
                    })
        
                    outer_loss += loss
                    
                meta_optim.zero_grad()
                outer_loss.div_(batch_size)
                outer_loss.backward()
        
            meta_optim.step()
        
            if step % args.val_freq == 0 and step != args.resume_step:
                if args.meta == 'MAML':
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
                print(f"step{step} model save to {path}")
            
            if args.use_scheduler:
                scheduler.step()
                
            step += 1
            pbar.update(1)
            
            if step > args.max_iters:
              break
        
    pbar.close()
    print(val_psnrs)
    

if __name__ == '__main__':
    main()