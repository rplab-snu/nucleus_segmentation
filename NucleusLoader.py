import torch
import torch.utils.data as data
import random
import numpy as np

import os
import scipy.misc
from glob import glob

import warnings
warnings.filterwarnings("ignore", ".*output shape of zoom.*")

class NSDataset(data.Dataset):
    # TODO : infer implementated
    def __init__(self, img_root, channel, sampler=None, infer=False, transform=None, torch_type="float", augmentation_rate=0.3):
        if type(img_root) == list:
            img_paths = [p for path in img_root for p in glob(path + "/*.npy")]
        else:
            img_paths = glob(img_root + '/*.npy')

        if len(img_paths) == 0:
            raise ValueError("Check data path : %s"%(img_root))

        self.origin_image_len = len(img_paths)
        self.img_paths = img_paths
        if transform is not None:
            self.img_paths += random.sample(img_paths, int(self.origin_image_len * augmentation_rate) )

        self.transform = [] if transform is None else transform
        self.torch_type = torch.float  if torch_type == "float" else torch.half

        self.channel = channel

    def __getitem__(self, idx):
        if self.channel == 1:
            return self._2D_image(idx)
        elif self.channel > 1:
            return self._25D_image(idx)
        else:
            raise ValueError("NSDataset data type must be 2d, 25d, 3d")

    def __len__(self):
        return len(self.img_paths)
    
    def _np2tensor(self, np):
        tmp = torch.from_numpy(np)
        return tmp.to(dtype=self.torch_type)

    def _2D_image(self, idx):
        img_path = self.img_paths[idx]
        img = np.load(img_path)
        # 2D ( 1 x H x W )
        input_np  = img[:, :448] 
        target_np = img[:, 448:]

        if idx >= self.origin_image_len:
            for t in self.transform:
                input_np, target_np = t(input_np, target_np)
        
        input_  = self._np2tensor(input_np ).resize_((1, *input_np.shape))
        target_ = self._np2tensor(target_np).resize_((1, *target_np.shape))
        return input_, target_, os.path.basename(img_path)

    def _25D_image(self, idx):
        img_path = self.img_paths[idx]
        img = np.load(img_path)

        input_np  = img[:, :448, :]
        target_np = img[:, 448:, 2:3]

        for t in self.transform:
            input_np, target_np = t([input_np, target_np])

        input_  = self._np2tensor(input_np ).permute(2, 0, 1)
        target_ = self._np2tensor(target_np).permute(2, 0, 1)

        return input_, target_, os.path.basename(img_path)

    
def make_weights_for_balanced_classes(seg_dataset):
    count = [0, 0] # No mask, mask
    for img, mask in seg_dataset:
        count[int((mask > 0).any())] += 1

    N = float(sum(count))
    weight_per_class = [N / c for c in count]

    weight = [0] * len(seg_dataset)
    for i, (img, mask) in enumerate(seg_dataset):
        weight[i] = weight_per_class[int((mask > 0).any())]

    return weight, count

def NucleusLoader(image_path, batch_size, patch_size=0, transform=None, sampler='',channel=1, torch_type="float", shuffle=True, cpus=1, infer=False, drop_last=True):
    dataset = NSDataset(image_path, channel, infer=infer, transform=transform, torch_type=torch_type)
    if sampler == "weight":
        weights, img_num_per_class = make_weights_for_balanced_classes(dataset)
        print("Sampler Weights : ", weights)
        weights = torch.DoubleTensor(weights)
        img_num_undersampling = img_num_per_class[1] * 2
        print("UnderSample to ", img_num_undersampling, " from ", img_num_per_class)
        sampler = data.sampler.WeightedRandomSampler(weights, img_num_undersampling)
        return data.DataLoader(dataset, batch_size, sampler=sampler,
                               shuffle=False, num_workers=cpus, drop_last=drop_last)
    
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=cpus, drop_last=drop_last)

def __test_npy(npy, txt):
    print(txt)
    print("shape : ", npy.shape)
    print("type : ", npy.dtype)
    print("min : ", npy.min())
    print("max : ", npy.max())
    

if __name__ == "__main__":
    fl_path  = "/data/00_Nuclues_segmentation/00_data/2D/Test_FL/"
    train_path  = "/data/00_Nuclues_segmentation/00_data/2D/New(50_Cells)/Only_Label/Train/"
    val_path  = "/data/00_Nuclues_segmentation/00_data/2D/New(50_Cells)/Only_Label/Val/"
    """
    train_npy = np.load(list(glob(train_path + "*.npy"))[0])
    val_npy   = np.load(list(glob(val_path + "*.npy"))[0])
    fl_npy    = np.load(list(glob(fl_path + "*.npy"))[0])

    __test_npy(train_npy, "train")
    __test_npy(val_npy, "val")
    __test_npy(fl_npy, "fl")
    """
    """
    npys = glob(fl_path + "*.npy")
    for path in npys:
        npy = np.load(path)
        save_path = "%s/%s"%("/home/rplab/workspace", "tmp")
        scipy.misc.imsave(save_path + ".jpg", npy[:, :448])
    """
    test_loader  = NucleusLoader(fl_path , 1, shuffle=False, drop_last=False)

    for i, (in_, ta_, fname) in enumerate(test_loader):
        print(fname)
        print(in_.shape)
        input_np = in_.numpy()[0, 0, :, :]
        save_path = "%s/%s"%("/home/rplab/workspace", fname[0][:-4])
        scipy.misc.imsave(save_path + ".jpg", input_np)
    


