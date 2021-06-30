import os,cv2,json,colorsys
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

class LSMI(data.Dataset):
    def __init__(self,root,split,image_pool,
                 input_type='uvl',output_type=None,
                 mask_black=None,mask_highlight=None,
                 data_augmentation=True):
        self.root = root                        # dataset root
        self.split = split                      # train / val / test
        self.image_pool = image_pool            # 1 / 2 / 3
        self.input_type = input_type            # uvl / rgb
        self.output_type = output_type          # None / illumination / uv
        self.mask_black = mask_black            # None or Masked value for black pixels
        self.mask_highlight = mask_highlight    # None or Saturation value
        self.data_augmentation = data_augmentation
        self.random_color = RandomColor(sat_min=0.2,sat_max=0.8,
                                        val_min=0.5,val_max=1,
                                        hue_threshold=0.2)

        self.image_list = sorted([f for f in os.listdir(os.path.join(root,split))
                                 if f.endswith(".tiff")
                                 and len(os.path.splitext(f)[0].split("_")[-1]) in image_pool])
        
        meta_file = os.path.join(self.root,'meta.json')
        with open(meta_file, 'r') as meta_json:
            self.meta_data = json.load(meta_json)

        print("[Data]\t"+str(self.__len__())+" "+split+" images are loaded from "+root)

    def __getitem__(self, idx):
        """
        Returns
        metadata        : meta information
        input_tensor    : input image (uvl or rgb)
        gt_tensor       : GT (None or illumination or chromaticity)
        mask            : mask for undetermined illuminations (black pixels) or saturated pixels
        """

        # parse fname
        fname = os.path.splitext(self.image_list[idx])[0]
        img_file = fname+".tiff"
        mixmap_file = fname+".npy"
        place, illum_count = fname.split('_')

        # 1. prepare meta information
        ret_dict = {}
        ret_dict["illum_chroma"] = []
        for illum_no in illum_count:
            illum_chroma = self.meta_data[place]["Light"+illum_no]
            ret_dict["illum_chroma"].append(illum_chroma)
        ret_dict["img_file"] = img_file
        ret_dict["place"] = place
        ret_dict["illum_count"] = illum_count

        # 2. prepare input & output GT
        # load mixture map & 3 channel RGB tiff image
        if len(illum_count) != 1:
            mixmap = np.load(os.path.join(self.root,self.split,fname+".npy")).astype('float32')
        else:
            mixmap = np.ones_like(input_rgb[:,:,0][:,:,None])
        input_path = os.path.join(self.root,self.split,img_file)
        input_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype('float32')
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

        # random data augmentation
        if self.data_augmentation:
            augment_chroma = self.random_color(len(illum_count))
            ret_dict["illum_chroma"] *= augment_chroma
            tint_map = self.mix_chroma(mixmap,augment_chroma)
            input_rgb = input_rgb * tint_map    # apply augmentation to input image

        # MCC chart masking
        if self.split == "train":
            mcc_mask = cv2.imread(os.path.join(self.root,self.split,place+"_mask.png"), cv2.IMREAD_GRAYSCALE)
            input_rgb = input_rgb * mcc_mask[:,:,None]

        # prepare input tensor
        ret_dict["input_rgb"] = self.totensor(input_rgb)
        ret_dict["input_uvl"] = self.totensor(self.rgb2uvl(input_rgb))

        # prepare output tensor
        illum_map = self.mix_chroma(mixmap,ret_dict["illum_chroma"])
        ret_dict["gt_illum"] = self.totensor(np.delete(illum_map, 1, axis=2))
        
        wb_matrix = np.ones_like(illum_map)
        wb_matrix[:,:,0] = illum_map[:,:,1] / illum_map[:,:,0]
        wb_matrix[:,:,2] = illum_map[:,:,1] / illum_map[:,:,2]
        output_rgb = input_rgb * wb_matrix
        ret_dict["gt_rgb"] = self.totensor(output_rgb)
        
        output_uvl = self.rgb2uvl(output_rgb)
        ret_dict["gt_uv"] = self.totensor(np.delete(output_uvl, 2, axis=2))

        # 3. prepare mask
        if self.split == 'train':
            mask = cv2.imread(os.path.join(self.root,self.split,place+"_mask.png"), cv2.IMREAD_GRAYSCALE)
            mask = mask[:,:,None].astype('float32')
        else:
            mask = np.ones_like(input_rgb[:,:,0], dtype='float32')[:,:,None]
        if self.mask_black != None:
            raise NotImplementedError("Implement black pixel masking!")
        if self.mask_highlight != None:
            raise NotImplementedError("Implement highlight masking!")
        mask = self.totensor(mask)
        ret_dict["mask"] = mask

        return ret_dict

    def mix_chroma(self, mixmap, chroma_list):
        ret = np.stack((np.zeros_like(mixmap[:,:,0]),)*3, axis=2)
        for i in range(len(chroma_list)):
            mixmap_3ch = np.stack((mixmap[:,:,i],)*3, axis=2)
            ret += (mixmap_3ch * [[chroma_list[i]]])
        
        return ret

    def rgb2uvl(self, img_rgb):
        epsilon = 1e-4
        img_uvl = np.zeros_like(img_rgb, dtype='float32')
        img_uvl[:,:,2] = np.log(img_rgb[:,:,1] + epsilon)
        img_uvl[:,:,0] = np.log(img_rgb[:,:,0] + epsilon) - img_uvl[:,:,2]
        img_uvl[:,:,1] = np.log(img_rgb[:,:,2] + epsilon) - img_uvl[:,:,2]

        return img_uvl

    def totensor(self, object):
        return torch.tensor(object).permute(2,0,1)

    def __len__(self):
        return len(self.image_list)

class RandomColor():
    def __init__(self,sat_min,sat_max,val_min,val_max,hue_threshold):
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.hue_threshold = hue_threshold

    def hsv2rgb(self,h,s,v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
    
    def threshold_test(self,hue_list,hue):
        if len(hue_list) == 0:
            return True
        for h in hue_list:
            if abs(h - hue) < self.hue_threshold:
                return False
        return True

    def __call__(self, illum_count):
        hue_list = []
        ret_chroma = []
        for i in range(illum_count):
            while(True):
                hue = np.random.uniform(0,1)
                saturation = np.random.uniform(self.sat_min,self.sat_max)
                value = np.random.uniform(self.val_min,self.val_max)
                chroma_rgb = np.array(self.hsv2rgb(hue,saturation,value), dtype='float32')
                chroma_rgb /= chroma_rgb[1]

                if self.threshold_test(hue_list,hue):
                    hue_list.append(hue)
                    ret_chroma.append(chroma_rgb)
                    break

        return np.array(ret_chroma)
        

def get_loader(config, split):
    dataset = LSMI(root=config.data_root,
                   split=split,
                   image_pool=config.image_pool,
                   input_type=config.input_type,
                   output_type=config.output_type,
                   mask_black=config.mask_black,
                   mask_highlight=config.mask_highlight,
                   data_augmentation=True)
    
    if split == 'test':
        dataloader = data.DataLoader(dataset,batch_size=1,shuffle=False,
                                     num_workers=config.num_workers)
    else:
        dataloader = data.DataLoader(dataset,batch_size=config.batch_size,
                                     shuffle=True,num_workers=config.num_workers)

    return dataloader

if __name__ == "__main__":
    
    train_set = LSMI(root='galaxy_256',
                      split='train',image_pool=[2],
                      input_type='uvl',output_type='illumination',
                      data_augmentation=True)

    train_loader = data.DataLoader(train_set, batch_size=2, shuffle=False)

    for batch in train_loader:
        print(batch["img_file"])
        print(batch["illum_chroma"].shape)
        print(batch["input"].shape)
        print(batch["output"].shape)
        print(batch["mask"].shape)

        input()