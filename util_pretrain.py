import torch
import matplotlib.pyplot as plt

def random_masking_for_loop(imgs, patch_size = 16, mask_ratio = 0.75):
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

def random_masking_vectorized(imgs, patch_size = 16, mask_ratio = 0.75):
    "Takes in the images and randomly masks them. Then it returns masked imgs"
    N, c, h, w = imgs.shape
    h_patch = h//patch_size   
    w_patch = w//patch_size   
    L = h_patch * w_patch     # total number of patches
    
    len_keep = int(L * (1 - mask_ratio))  # number of patches to keep of the total

    noise = torch.rand(N, L, device=imgs.device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.zeros([N, L], device=imgs.device)
    mask[:, :len_keep] = 1.
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)   # [bs, (h_patch x w_patch)]

    # resize mask
    mask_3d = torch.reshape(mask, shape = (N, h_patch, w_patch))
    
    # add the color channels to mask tensor
    mask_3d = mask_3d[:, None, :, :]
    mask_colored = mask_3d.repeat(1, 3, 1, 1)

    mask_colored_increased = torch.repeat_interleave(mask_colored, patch_size, dim=2)
    batch_mask = torch.repeat_interleave(mask_colored_increased, patch_size, dim=3)

    masked_imgs = imgs * batch_mask
    return masked_imgs, batch_mask


def forward_loss(imgs, preds, masks):
    """
    inputs:
    imgs --> non-masked images of shape bs, c, h, w
    preds --> predictions of shape bs, c, h, w
    masks --> masks of shape bs, c, h, w --> 1 means keep and 0 means discard
    (add something more if you want)

    outputs:
    removed_patch_loss --> the loss between only the masked parts
    simple_loss --> the loss between all the pixels
    (add something more if you want)
    """
    mse_all_loss = (imgs - preds)**2
    inverse_masks = 1 - masks # 0 is keep (non-masked parts), 1 is discard (masked parts)
    removed_patch_loss = (mse_all_loss * inverse_masks).sum() / inverse_masks.sum()
    simple_loss = torch.mean(mse_all_loss)
    return removed_patch_loss, simple_loss



def show_images(non_masked, masked, pred, sample, epoch):
    non_masked_np = non_masked[sample].permute(1, 2, 0).cpu().numpy()
    masked_np = masked[sample].permute(1, 2, 0).cpu().numpy()
    
    pred_detached = pred.detach()
    pred_np = pred_detached[sample].permute(1, 2, 0).cpu().numpy()
    
    plt.imshow(non_masked_np)
    plt.savefig('original_img' + '_' + str(sample) + '_ep' + str(epoch) + '.png')

    plt.imshow(masked_np)
    plt.savefig('masked_img' + '_' + str(sample) + '_ep' + str(epoch) + '.png')

    plt.imshow(pred_np)
    plt.savefig('predicted_img' + '_' + str(sample) + '_ep' + str(epoch) + '.png')
    return




