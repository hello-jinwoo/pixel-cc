import math,cv2,torch
import numpy as np

from skimage.metrics import structural_similarity as ssim

# def get_AE(pred, GT, mask):
#     """
#     pred & GT   : (b,h,w,c) numpy array
#                   These array contains 2 channel values which means R,B

#     mask        : MCC chart area + Black pixels (train)
#                   Black pixels (test)

#     returns     : Mean Angular Error between pred & GT
#     """
#     b,h,w,c = pred.shape

#     pred = pred.reshape((-1,c))
#     GT = GT.reshape((-1,c))
#     mask = mask.reshape((-1))

#     pred_rgb = np.vstack([pred[:, 0], np.ones_like(pred[:, 0]), pred[:, 1]])
#     GT_rgb = np.vstack([GT[:, 0], np.ones_like(GT[:, 0]), GT[:, 1]])

#     pred_norm = pred_rgb / (np.linalg.norm(pred_rgb, axis=0) + 1e-8)
#     GT_norm = GT_rgb / (np.linalg.norm(GT_rgb, axis=0) + 1e-8)

#     dot_product = np.clip(np.sum(pred_norm * GT_norm, axis=0), -1, 1)
#     angle = np.arccos(dot_product) * mask

#     MAE_rad = np.sum(angle) / np.sum(mask)
#     MAE_deg = math.degrees(MAE_rad)

#     return MAE_deg

# def get_chroma_MAE(pred,gt,camera=None,mask=None):
#     """
#     pred : (b,c,h,w)
#     gt : (b,c,h,w)
#     """
#     if camera == 'galaxy':
#         pred = torch.clamp(pred, 0, 1023)
#         gt = torch.clamp(gt, 0, 1023)
#     elif camera == 'sony' or camera == 'nikon':
#         pred = torch.clamp(pred, 0, 16383)
#         gt = torch.clamp(gt, 0, 16383)

#     b,c,h,w, = pred.shape

#     pred = torch.exp(pred.reshape((b,c,-1)))
#     gt = torch.exp(gt.reshape((b,c,-1)))
#     ones = torch.unsqueeze(torch.ones_like(pred[:,0,:]), 1)
    
#     pred = torch.cat((pred,ones), 1)
#     gt = torch.cat((gt,ones), 1)
    
#     pred_norm = pred / (torch.unsqueeze(torch.linalg.norm(pred, dim=1), 1) + 1e-8)
#     GT_norm = gt / (torch.unsqueeze(torch.linalg.norm(gt, dim=1), 1) + 1e-8)

#     dot_product = torch.clip(torch.sum(pred_norm * GT_norm, dim=1), -1, 1)
#     radian = torch.acos(dot_product)
#     degree = torch.rad2deg(radian)

#     MAE_img = torch.mean(degree, dim=1)
#     MAE_batch = torch.mean(MAE_img)

#     return MAE_batch.item()

def get_MAE(pred,gt,tensor_type,camera=None,mask=None):
    """
    pred : (b,c,w,h)
    gt : (b,c,w,h)
    """
    if tensor_type == "rgb":
        if camera == 'galaxy':
            pred = torch.clamp(pred, 0, 1023)
            gt = torch.clamp(gt, 0, 1023)
        elif camera == 'sony' or camera == 'nikon':
            pred = torch.clamp(pred, 0, 16383)
            gt = torch.clamp(gt, 0, 16383)

    pred = pred.permute(0,2,3,1).reshape((-1,3))
    gt = gt.permute(0,2,3,1).reshape((-1,3))
    mask = mask.permute(0,2,3,1)

    pred_norm = torch.linalg.norm(pred,dim=-1,keepdim=True) + 1e-8
    gt_norm = torch.linalg.norm(gt,dim=-1,keepdim=True) + 1e-8

    pred_unit = pred / pred_norm
    gt_unit = gt / gt_norm

    cos_similarity = (pred_unit * gt_unit).sum(dim=-1)
    cos_similarity = torch.clip(cos_similarity,0,1)

    ang_error = torch.rad2deg(torch.acos(cos_similarity))

    if mask is not None:
        mask_reshaped = torch.reshape(mask, (-1,))
        ang_error = ang_error[mask_reshaped!=0]
    mean_angular_error = ang_error.mean()

    return mean_angular_error


def get_PSNR(pred, gt, white_level):
    """
    pred & gt   : (b,c,h,w) numpy array 3 channel RGB
    returns     : average PSNR of two images
    """
    pred = pred.permute(0,2,3,1).numpy()
    gt = gt.permute(0,2,3,1).numpy()

    if white_level != None:
        pred = np.clip(pred, 0, white_level)
        gt = np.clip(gt, 0, white_level)

    return cv2.PSNR(pred, gt, white_level)

def get_SSIM(pred, GT, white_level):
    """
    pred & GT   : (h,w,c) numpy array 3 channel RGB

    returns     : average PSNR of two images
    """
    if white_level != None:
        pred = np.clip(pred, 0, white_level)
        GT = np.clip(GT, 0, white_level)

    return ssim(pred, GT, multichannel=True, data_range=white_level)

# def apply_wb(I_patch, pred, output_mode):
#     pred_patch = torch.zeros_like(I_patch) # b,c,h,w
    
#     if output_mode == 'uv':
#         pred_patch[:,1,:,:] = I_patch[:,1,:,:]
#         pred_patch[:,0,:,:] = I_patch[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
#         pred_patch[:,2,:,:] = I_patch[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)

#     elif output_mode == 'illumination':
#         pred_patch[:,1,:,:] = I_patch[:,1,:,:]
#         pred_patch[:,0,:,:] = I_patch[:,0,:,:] * (1 / (pred[:,0,:,:]+1e-8))    # R = R * (1/R_i)
#         pred_patch[:,2,:,:] = I_patch[:,2,:,:] * (1 / (pred[:,1,:,:]+1e-8))    # B = B * (1/B_i)

#     return pred_patch