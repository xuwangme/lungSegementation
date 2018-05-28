"""
# @author: xuwang
# @function: 搭建UNet网络
# @date: 2018/5/8 12:51
"""
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D

class UNetModel(object):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def buildUNet(self, input_shape):
        merge_axis = -1  # Feature maps are concatenated along last axis (for tf backend)
        input_data = Input(shape=input_shape)
        conv1 = Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(input_data)
        conv1 = Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(pool1)
        conv2 = Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(pool2)
        conv3 = Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(pool3)
        conv4 = Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(filters=256, kernel_size=self.kernel_size, padding='same', activation='relu')(pool4)

        up1 = UpSampling2D(size=(2, 2))(conv5)
        conv6 = Conv2D(filters=256, kernel_size=self.kernel_size, padding='same', activation='relu')(up1)
        conv6 = Conv2D(filters=256, kernel_size=self.kernel_size, padding='same', activation='relu')(conv6)
        merged1 = concatenate([conv4, conv6], axis=merge_axis)
        conv6 = Conv2D(filters=256, kernel_size=self.kernel_size, padding='same', activation='relu')(merged1)

        up2 = UpSampling2D(size=(2, 2))(conv6)
        conv7 = Conv2D(filters=256, kernel_size=self.kernel_size, padding='same', activation='relu')(up2)
        conv7 = Conv2D(filters=256, kernel_size=self.kernel_size, padding='same', activation='relu')(conv7)
        merged2 = concatenate([conv3, conv7], axis=merge_axis)
        conv7 = Conv2D(filters=256, kernel_size=self.kernel_size, padding='same', activation='relu')(merged2)

        up3 = UpSampling2D(size=(2, 2))(conv7)
        conv8 = Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(up3)
        conv8 = Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(conv8)
        merged3 = concatenate([conv2, conv8], axis=merge_axis)
        conv8 = Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', activation='relu')(merged3)

        up4 = UpSampling2D(size=(2, 2))(conv8)
        conv9 = Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(up4)
        conv9 = Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(conv9)
        merged4 = concatenate([conv1, conv9], axis=merge_axis)
        conv9 = Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', activation='relu')(merged4)

        conv10 = Conv2D(filters=1, kernel_size=self.kernel_size, padding='same', activation='sigmoid')(conv9)

        output_data = conv10
        UNet = Model(input_data, output_data)
        return UNet

