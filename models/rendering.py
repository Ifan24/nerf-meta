import torch
import collections

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields

def get_d (rays_d, n_sample):
  """
    input: 
      rays_d: [B, 3] batch of rays direction
      n_sample: N, number of sample point in each ray
    output:
      viewdirs: [B, N, 3] direction for each sample point for that batch
  """

  viewdirs = rays_d
  # [B, 3]
  # repeat for each sample point
  viewdirs = torch.repeat_interleave(viewdirs[...,None,:], n_sample, dim=-2)
  # Make all directions unit magnitude.
  viewdirs = viewdirs / torch.linalg.norm(viewdirs, axis=-1, keepdims=True)
  return viewdirs
  
def get_rays_shapenet(hwf, poses):
    """
    shapenet camera intrinsics are defined by H, W and focal.
    this function can handle multiple camera poses at a time.
    Args:
        hwf (3,): H, W, focal
        poses (N, 4, 4): pose for N number of images
        
    Returns:
        rays_o (N, H, W, 3): ray origins
        rays_d (N, H, W, 3): ray directions
    """
    if poses.ndim == 2:
        poses = poses.unsqueeze(dim=0)  # if poses has shape (4, 4)
                                        # make it (1, 4, 4)

    H, W, focal = hwf
    yy, xx = torch.meshgrid(torch.arange(0., H, device=focal.device),
                            torch.arange(0., W, device=focal.device))
    direction = torch.stack([(xx-0.5*W)/focal, -(yy-0.5*H)/focal, -torch.ones_like(xx)], dim=-1) # (H, W, 3)
                                        
    rays_d = torch.einsum("hwc, nrc -> nhwr", direction, poses[:, :3, :3]) # (N, H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    rays_o = poses[:, :3, -1] # (N, 3)
    rays_o = rays_o[:, None, None, :].expand_as(rays_d) # (N, H, W, 3)

    return rays_o, rays_d

def get_rays_shapenet_mipNerf(hwf, poses):
    """
    shapenet camera intrinsics are defined by H, W and focal.
    this function can handle multiple camera poses at a time.
    Args:
        hwf (3,): H, W, focal
        poses (N, 4, 4): pose for N number of images
        
    Returns:
        rays: len(List) = N
            origins:     (H, W, 3)
            directions:  (H, W, 3)
            viewdirs:    (H, W, 3)
            radii:       (H, W, 1)
            lossmult:    (H, W, 1)
            near:        (H, W, 1)
            far:         (H, W, 1)
    """
    
    h, w, focal = hwf
    near = 2
    far = 6

    if poses.ndim == 2:
        poses = poses.unsqueeze(dim=0)

    H, W, focal = hwf
    yy, xx = torch.meshgrid(torch.arange(0., H, device=focal.device),
                            torch.arange(0., W, device=focal.device))
    direction = torch.stack([(xx-0.5*W)/focal, -(yy-0.5*H)/focal, -torch.ones_like(xx)], dim=-1) # (H, W, 3)

    directions = [(direction @ c2w[:3, :3].T).clone() for c2w in poses]

    origins = [
        torch.broadcast_to(c2w[:3, -1], v.shape).clone()
        for v, c2w in zip(directions, poses)
    ]
    viewdirs = [
        v / torch.linalg.norm(v, axis=-1, keepdims=True) for v in directions
    ]

    def broadcast_scalar_attribute(x):
        return [
            x * torch.ones_like(origins[i][..., :1])
            for i in range(len(poses))
        ]

    lossmults = broadcast_scalar_attribute(1).copy()
    nears = broadcast_scalar_attribute(near).copy()
    fars = broadcast_scalar_attribute(far).copy()

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = [
        torch.sqrt(torch.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
    ]
    dx = [torch.cat([v, v[-2:-1, :]], 0) for v in dx]
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = [v[..., None] * 2 / torch.sqrt(torch.tensor(12)) for v in dx]

    rays = Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=lossmults,
        near=nears,
        far=fars)
    del origins, directions, viewdirs, radii, lossmults, nears, fars, direction
    
    return rays

Rays_with_color = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', "color"))
Rays_with_color_keys = Rays._fields

def get_raybatch(rays, imgs, chunk_size=4096):
    """
      input:
        rays: len(List) = N
          origins:     (H, W, 3)
          ...

        imgs: tensor (N, H, W, 3)

      output:
        Rays_with_color: len(List) = N*H*W/chunk_size
          origins: (chunk_size, 3)
          ...
          color: (chunk_size, 3)
        
        val_mask: List(Tensor(H, W, 3)) N

    """

    # change Rays to list: [[origins], [directions], [viewdirs], [radii], [lossmult], [near], [far]]
    single_image_rays = [getattr(rays, key) for key in Rays_keys]
    val_mask = single_image_rays[-3]

    # flatten each Rays attribute and put on device
    single_image_rays = [torch.stack(rays_attr).reshape(-1, rays_attr[0].shape[-1]) for rays_attr in single_image_rays]
    single_image_rays.append(imgs.reshape(-1, 3))

    # single_image_rays = [[N*H*W, 3], ..., [N*H*W, 1]]
    # get the amount of full rays of an image
    length = single_image_rays[0].shape[0]
    # length = N*H*W

    # divide each Rays attr into N groups according to chunk_size,
    # the length of the last group <= chunk_size
    single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for
                         rays_attr in single_image_rays]
    # get N, the N for each Rays attr is the same
    length = len(single_image_rays[0])
    # length = N*H*W / chunk_size 
    # generate N Rays instances
    single_image_rays = [Rays_with_color(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]
    return single_image_rays, val_mask
    
        
def get_rays_tourism(H, W, kinv, pose):
    """
    phototourism camera intrinsics are defined by H, W and kinv.
    Args:
        H: image height
        W: image width
        kinv (3, 3): inverse of camera intrinsic
        pose (4, 4): camera extrinsic
    Returns:
        rays_o (H, W, 3): ray origins
        rays_d (H, W, 3): ray directions
    """
    yy, xx = torch.meshgrid(torch.arange(0., H, device=kinv.device),
                            torch.arange(0., W, device=kinv.device))
    pixco = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
 
    directions = torch.matmul(pixco, kinv.T) # (H, W, 3)

    rays_d = torch.matmul(directions, pose[:3, :3].T)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # (H, W, 3)
    
    rays_o = pose[:3, -1].expand_as(rays_d) # (H, W, 3)

    return rays_o, rays_d
    

def sample_points(rays_o, rays_d, near, far, num_samples, perturb=False):
    """
    Sample points along the ray
    Args:
        rays_o (num_rays, 3): ray origins
        rays_d (num_rays, 3): ray directions
        near (float): near plane
        far (float): far plane
        num_samples (int): number of points to sample along each ray
        perturb (bool): if True, use randomized stratified sampling
    Returns:
        t_vals (num_rays, num_samples): sampled t values
        coords (num_rays, num_samples, 3): coordinate of the sampled points
    """
    num_rays = rays_o.shape[0]

    t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)
    t_vals = t_vals.expand(num_rays, num_samples)   # t_vals has shape (num_samples)
                                                    # we must broadcast it to (num_rays, num_samples)
    if perturb:
        rand = torch.rand_like(t_vals) * (far-near)/num_samples
        t_vals = t_vals + rand

    coords = rays_o.unsqueeze(dim=-2) + t_vals.unsqueeze(dim=-1) * rays_d.unsqueeze(dim=-2)

    return t_vals, coords


def volume_render(rgbs, sigmas, t_vals, white_bkgd=False):
    """
    Volume rendering function.
    Args:
        rgbs (num_rays, num_samples, 3): colors
        sigmas (num_rays, num_samples): densities
        t_vals (num_rays, num_samples): sampled t values
        white_bkgd (bool): if True, assume white background
    Returns:
        color (num_rays, 3): color of the ray
    """
    # for phototourism, final delta is infinity to capture background
    # https://github.com/tancik/learnit/issues/4
    
    if white_bkgd:
        bkgd = 1e-3
    else:
        bkgd = 1e10

    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    delta_final = bkgd * torch.ones_like(deltas[:, -1:])
    deltas = torch.cat([deltas, delta_final], dim=-1) # (num_rays, num_samples)

    alphas = 1 - torch.exp(-deltas*sigmas)
    transparencies = torch.cat([
                                torch.ones_like(alphas[:, :1]),
                                torch.cumprod(1 - alphas[:, :-1] + 1e-10, dim=-1)
                                ], dim=-1)
    weights = alphas * transparencies # (num_rays, num_samples)
    
    # color = torch.sum(rgbs*weights.unsqueeze(-1), dim=-2)
    color = torch.einsum("rsc, rs -> rc", rgbs, weights)
    
    if white_bkgd:
        # composite the image to a white background
        color = color + 1 - weights.sum(dim=-1, keepdim=True)

    return color
