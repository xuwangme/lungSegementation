"""
# @author: xuwang
# @function:
# @date: 2018/5/7 11:15
"""
import os
import numpy as np
from skimage import io, exposure

class Preprocess(object):
    def __init__(self):
        self.raw_dir= "./data/raw_data/All247images"
        self.left_mask_dir = "./data/raw_data/raw_masks/left lung"
        self.right_mask_dir = "./data/raw_data/raw_masks/right lung"
        self.image_dir = "./data/images"
        self.mask_dir = "./data/masks"

    def generateImage(self):
        print(" ---------------->>>>>>> start generate images")
        for filename in os.listdir(self.raw_dir):
            image = 1. - np.fromfile(os.path.join(self.raw_dir, filename), dtype='>u2').reshape((2048, 2048)) * 1. / 4096
            # 直方图均衡化
            image = exposure.equalize_hist(image)
            io.imsave(os.path.join(self.image_dir, filename[:-4] + '.png'), image)
        print(" ---------------->>>>>>> complete generate images")

    def generateMask(self):
        print(" ---------------->>>>>>> start generate masks")
        for filename in os.listdir(self.raw_dir):
            left_lung = io.imread(os.path.join(self.left_mask_dir, filename[:-4] + ".gif"))
            right_lung = io.imread(os.path.join(self.right_mask_dir, filename[:-4] + ".gif"))
            io.imsave(os.path.join(self.mask_dir, filename[:-4] + '_mask.png'), np.clip(left_lung + right_lung, 0, 255))
        print(" ---------------->>>>>>> complete generate masks")
