import os
from glob import glob

import torch.utils.data as data
import torchvision.transforms as transforms

class MARDataset(data.Dataset):
    def __init__(self, image_path, infer, image_type):
        self.image_type = image_type
        self.imgs = glob(os.path.join(image_path, "*.npy"))        
        self.__getitem__ = self._load_to_tensor if infer else self._train_getitem
    
    def _load_to_tensor(self, idx):
        path = self.img[idx]
        img = np.load(path)
        img = torch.from_numpy(img).float()
        img = img.reshape(1, *img.shape)

        if self.image_type == "sin2res":
            img /= 100
        else:
            img /= 4095
        return img, os.path.basename(path)

    def _train_getitem(self, idx):
        img, f_name = self._load_to_tensor(idx)
        input_  = img[:, :, :512]
        target_ = img[:, :, 512:]
        return input_, target_, f_name

    def __len__(self):
        return len(self.imgs)

def MARLoader(image_path, batch_size, image_type, cpus=1, infer=False):
    dataset = MARDataset(image_path, infer, image_type)
    return data.DataLoader(dataset, batch_size, shuffle=True, num_workers=cpus, drop_last=not infer)
