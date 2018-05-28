"""
# @author: xuwang
# @function:
# @date: 2018/5/14 13:14
"""
import os
import numpy as np
import matplotlib.pyplot as plt
class Evaluate(object):
    def __init__(self):
        self.mask_test_path = "./data/gt_masks/gt_mask.npy"
        self.predict_mask_path = "./data/pre_masks/pre_mask.npy"
        self.image_shape = (256, 256)

    def eval(self):
        # pre_names = os.listdir(self.pre_dir)
        # gt_names = os.listdir(self.gt_dir)
        # for pre_name in pre_names:
        #     prefix_name = pre_name[:-8]
        #     gt_name = pre_name + "_mask.png"
        #     pre_mask = io.imread(os.path.join(self.pre_dir, pre_name))
        #     pre_mask = transform.resize(pre_mask, self.image_shape)
        #     gt_mask = io.imread(os.path.join(self.gt_dir, gt_name))
        #     gt_mask = transform.resize(gt_mask, self.image_shape)
        gt_masks = np.load(self.mask_test_path)
        pre_masks = np.load(self.predict_mask_path)
        gt_masks = np.squeeze(gt_masks, axis=-1)
        pre_masks = np.squeeze(pre_masks, axis=-1)
        dice_values = []
        for i in range(gt_masks.shape[0]):
            gt_mask = gt_masks[i]
            pre_mask = pre_masks[i]
            # 0-1标准化
            pre_mask = (pre_mask - pre_mask.min()) / (pre_mask.max() - pre_mask.min())
            dice_value = self.calcDice(pre_mask, gt_mask)
            dice_values.append(dice_value)
        print("min dice_value: ", min(dice_values))
        print("max dice_value: ", max(dice_values))
        print("mean dice_value: ", (sum(dice_values) / len(dice_values)))

    def calcDice(self, pre_mask, gt_mask):
        pre_mask = pre_mask > 0.5
        gt_mask = gt_mask > 0.5
        intersection = np.logical_and(pre_mask, gt_mask).sum()
        # 拉普拉斯平滑处理，并求Dice
        dice_value = (2 * intersection) / (pre_mask.sum() + gt_mask.sum())
        return dice_value

