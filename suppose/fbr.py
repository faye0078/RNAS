# feature maps by different resolutions
import sys
import numpy as np
import glob
sys.path.append('/media/dell/DATA/wy/RNAS')
import torch
from PIL import Image
from model.seg_hrnet import get_seg_model
from engine.baseEngine import BaseEngine
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        
if __name__ == '__main__':
    model = get_seg_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = BaseEngine(model, optimizer, loss_fn, device)
    engine.model.eval()
    model_paths = '/media/dell/DATA/wy/RNAS/model/weights/hrnet_512.pth.tar'
    engine.load(model_paths)
    
    dir_path = "/media/dell/DATA/wy/data/GID-5/4096/image/GF2_PMS2__L1A0001119060-MSS2/"
    images = glob.glob(dir_path + '*.tif')
    # image_path = "/media/dell/DATA/wy/data/Large-scale Classification_5classes/image_NirRGB/GF2_PMS2__L1A0001119060-MSS2.tif"
    for image_path in images:
        image = np.asarray(Image.open(image_path), dtype=np.uint8)
        scale = 1.0 / 255,  # SCALE
        mean = np.array([0.485, 0.456, 0.406, 0.411]).reshape((1, 1, 4)),  # MEAN
        std = np.array([0.229, 0.224, 0.225, 0.227]).reshape((1, 1, 4)),  # STD
        image = (scale * image - mean) / std
        
        image = torch.from_numpy(image.transpose((0, 3, 1, 2))).float().cuda()
        image_path = os.path.join("/media/dell/DATA/wy/RNAS/images", dir_path.split('/')[-2] + '_' + dir_path.split('/')[-4], image_path.split('/')[-1])
        dir = os.path.dirname(image_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        engine.model.eval()
        with torch.no_grad():
            result = engine.model(image, image_path)
        torch.cuda.empty_cache()
        print(result.shape)
    
    