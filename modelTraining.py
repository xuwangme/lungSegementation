"""
# @author: xuwang
# @function: 模型训练
# @date: 2018/5/8 10:13
"""
import os
import pandas as pd
import numpy as np
from skimage import transform, io
from model import UNetModel

class ModelTraining(object):
    def __init__(self):
        self.csv_path = "./data/image_name.csv"
        self.processed_data_dir = "./data/processed_data"
        self.rate = 0.2
    def loadData(self):
        csv_file = pd.read_csv(self.csv_path)
        # 对样本顺序进行打乱
        csv_file = csv_file.sample(frac=1, random_state=24)
        image_shape = (256, 256)
        image_list = []
        mask_list = []
        for i, obj in csv_file.iterrows():
            image = io.imread(os.path.join(self.processed_data_dir, obj[0]))
            image = transform.resize(image, image_shape)
            image = np.expand_dims(image, -1)
            mask = io.imread(os.path.join(self.processed_data_dir, obj[1]))
            mask = transform.resize(mask, image_shape)
            mask = np.expand_dims(mask, -1)
            image_list.append(image)
            mask_list.append(mask)
        image_array = np.array(image_list)
        # 对image数据进行z-score 标准化
        image_array -= image_array.mean()
        image_array /= image_array.std()

        mask_array = np.array(mask_list)

        return image_array, mask_array

    def train(self, seg):
        image_array, mask_array = self.loadData()
        start_loca = int(image_array.shape[0] * seg)
        test_size = int(image_array.shape[0] * self.rate)
        image_train = image_array[start_loca:(start_loca+test_size)]
        image_test = np.concatenate((image_array[:start_loca], image_array[(start_loca+test_size):]), axis=0)
        mask_train = mask_array[start_loca:(start_loca+test_size)]
        mask_test = np.concatenate((mask_array[:start_loca], mask_array[(start_loca+test_size):]), axis=0)
        np.save("./gt_masks/gt_mask_" + str(seg), mask_test)
        input_shape = image_array[0].shape
        unet_model = UNetModel()
        UNet = unet_model.buildUNet(input_shape)
        UNet.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        UNet.fit(image_train, mask_train, batch_size=4, nb_epoch=20, verbose=1, validation_split=0.1, shuffle=True)
        loss, acc = UNet.evaluate(image_test, mask_test)
        print("loss: ", loss)
        print("accuracy: ", acc)
        predict_mask = UNet.predict(image_test, verbose=1)
        print("predict done !")
        print(type(predict_mask))
        np.save("./pre_masks/pre_mask_" + str(seg), predict_mask)
