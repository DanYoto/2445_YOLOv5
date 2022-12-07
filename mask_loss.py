import torch

def random_masking(imgs, patch_size = 16, mask_ratio = 0.75):
    # Take in a batch of images and mask it with 'patch_size' big patches to
    # a ratio of 'mask_ratio'
    bs, c, h, w = imgs.shape

    h_patch = h // patch_size   # number of patches in heights
    w_patch = w // patch_size   # number of patches in width
    
    masked = torch.zeros(bs, h_patch * w_patch, device = imgs.device)   
    nr_zeros = h_patch * w_patch 
    for i in range(bs):  
        # Create an unique mask for each image in the batch to increase the learning
        zeros = torch.zeros(nr_zeros)   
        zeros[:int((1-mask_ratio)*nr_zeros)] = 1. # 1 indication patches to keep and 0 patches to discard. 
    
        # randomize the 1's and 0's
        indicies = torch.randperm(nr_zeros)  
        masked[i] = zeros[indicies] 

    mask_tensor = torch.reshape(masked, (bs, h_patch, w_patch))
    
    # Add a new channel
    channel_1 = torch.unsqueeze(mask_tensor, dim=1)    # bs, c=1, h_patch, w_patch
    channel_3 = channel_1.repeat(1, 3, 1, 1)           # bs, c=3, h_patch, w_patch (each channel having the same masking)

    # interpolate each number to a patch_size to obtain a mask with shape bs, c, h, w
    mask = torch.nn.functional.interpolate(channel_3, scale_factor = patch_size, mode='nearest')

    # multiply image with mask
    masked_imgs = imgs * mask
    return masked_imgs, mask

def mask_loss(imgs, preds, masks):
    """
    inputs:
    imgs --> non-masked images of shape bs, c, h, w
    preds --> predictions of shape bs, c, h, w
    masks --> masks of shape bs, c, h, w --> 1 means keep and 0 means discard
    (add something more if you want)
    """
    mse_all_loss = (imgs - pred)**2
    removed_patch_loss = (mse_all_loss * masks).sum() / masks.sum()
    simple_loss = torch.mean(mse_all_loss)
    """
    outputs:
    removed_patch_loss --> the loss between only the masked parts
    simple_loss --> the loss between all the pixels
    (add something more if you want)
    """


    return removed_patch_loss, simple_loss





