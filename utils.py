import torch

def apply_wb(org_img,pred,pred_type):
    pred_rgb = torch.zeros_like(org_img) # b,c,h,w

    if pred_type == "illumination":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,0,:,:] * (1 / (pred[:,0,:,:]+1e-8))    # R_wb = R * (1/illum_R)
        pred_rgb[:,2,:,:] = org_img[:,2,:,:] * (1 / (pred[:,2,:,:]+1e-8))    # B_wb = B * (1/illum_B)
    elif pred_type == "uv":
        pred_rgb[:,1,:,:] = org_img[:,1,:,:]
        pred_rgb[:,0,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
        pred_rgb[:,2,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)
    
    return pred_rgb