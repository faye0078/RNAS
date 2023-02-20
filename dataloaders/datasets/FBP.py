import os
import numpy as np
import random
from transforms import *
from PIL import Image
from torch.utils.data import Dataset


class FBPDataset(Dataset):

    def __init__(self, image_size_rand, data_file, data_dir, transform_trn=None, transform_val=None, transform_test=None):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        try:
            self.datalist = [
                (k[0], k[1])
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
        except ValueError:  # Adhoc for test.
            self.datalist = [
                (k, k) for k in map(lambda x: x.decode("utf-8").strip("\n"), datalist)
            ]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.image_size_rand = image_size_rand
        
        self.img_idx = 0
        
        self.image = None
        self.mask = None
        self.image_size = random.randint(image_size_rand[0], image_size_rand[1])
        
        self.i = 0
        self.j = 0

    def set_config(self, crop_size, resize_side):
        self.transform_trn.transforms[0].resize_side = resize_side
        self.transform_trn.transforms[2].crop_size = crop_size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self):

        image, mask = self._get_pach()

        sample = {"image": image, "mask": mask}
        if self.stage == "train":
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == "val":
            if self.transform_val:
                sample = self.transform_val(sample)
        elif self.stage == 'test':
            if self.transform_test:
                sample = self.transform_test(sample)
        return sample
    
    def _get_pach(self):
        if self.image is None:
            self._refresh_img()
            
        if self.i + self.image_size > self.image.shape[0]:
            self.i = 0
            self.j += self.image_size
            if self.j + self.image_size > self.image.shape[1]:
                self._refresh_img()
                self.j = 0
                
        image = self.image[self.i:self.i+self.image_size, self.j:self.j+self.image_size]
        mask = self.mask[self.i:self.i+self.image_size, self.j:self.j+self.image_size]
        
        self.i += self.image_size
        
        return image, mask
        
    def _refresh_img(self):
        self.image = np.asarray(Image.open(os.path.join(self.root_dir, self.datalist[self.img_idx][0])))
        self.mask = np.asarray(Image.open(os.path.join(self.root_dir, self.datalist[self.img_idx][1])))
        self.i = 0
        self.j = 0
        
