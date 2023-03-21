import os
import numpy as np
import random
from dataloaders.datasets.transforms import *
from PIL import Image
from torch.utils.data import Dataset


class FBPDataset(Dataset):

    def __init__(self, stage, image_size_rand, data_file, data_dir, transform_trn=None, transform_val=None, transform_test=None):
        self.stage = stage
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
        self.index_list = None
        
        self.image = None
        self.mask = None
        self.image_size = random.randint(image_size_rand[0], image_size_rand[1])
        print("image_size: ", self.image_size)

        self._refresh_img()
        
    def update_image_size(self):
        self.image_size = random.randint(self.image_size_rand[0], self.image_size_rand[1])
        self.img_idx = 0
        print("image_size: ", self.image_size)
        
    def _caculate_patch_num(self):
        num_x = int(np.ceil(self.image.shape[0] / self.image_size))
        num_y = int(np.ceil(self.image.shape[1] / self.image_size))
        return num_x * num_y
    
    def __len__(self):
        return len(self.datalist) * self._caculate_patch_num()
    
    def _refresh_img(self):
        self.image = np.asarray(Image.open(os.path.join(self.root_dir, self.datalist[self.img_idx][0])))
        self.mask = np.asarray(Image.open(os.path.join(self.root_dir, self.datalist[self.img_idx][1])))
        self._make_patch_index()
        self.img_idx += 1
        
    def _make_patch_index(self):
        index_list = []
        num_x = int(np.ceil(self.image.shape[0] / self.image_size))
        num_y = int(np.ceil(self.image.shape[1] / self.image_size))
        for i in range(num_x):
            for j in range(num_y):
                if i == num_x - 1:
                    x_index = self.image.shape[0] - self.image_size
                else:
                    x_index = i * self.image_size
                if j == num_y - 1:
                    y_index = self.image.shape[1] - self.image_size
                else:
                    y_index = j * self.image_size
                    
                patch_index = (x_index, y_index)
                index_list.append(patch_index)
        # 打乱顺序
        index_list = random.sample(index_list, len(index_list))
        
        if len(index_list) != self._caculate_patch_num():
            exit('error: index_list length is not equal to patch num')
            
        self.index_list = index_list
        
    def _get_pach(self):
        index = self._get_patch_index()
        image = self.image[index[0]:index[0]+self.image_size, index[1]:index[1]+self.image_size]
        mask = self.mask[index[0]:index[0]+self.image_size, index[1]:index[1]+self.image_size]        
        return image, mask
    
    def _get_patch_index(self):
        if len(self.index_list) == 0:
            self._refresh_img()
        return self.index_list.pop()

    def __getitem__(self, index):
        
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
    
    
        
    
        
        
