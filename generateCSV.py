"""
# @author: xuwang
# @function: 生成CSV文件
# @date: 2018/5/7 15:23
"""
import os

def generateCSV(image_dir, mask_dir, csvfile_path):
    csv_file = open(csvfile_path, "w+")
    image_names = os.listdir(image_dir)
    mask_names = os.listdir(mask_dir)
    csv_file.write("%s%s" % ("image,mask", "\n"))
    for i in range(len(image_names)):
        csv_file.write("%s%s" % (image_names[i] + "," + mask_names[i], "\n"))
    csv_file.close()

if __name__ == '__main__':
    image_dir = "./data/images"
    mask_dir = "./data/masks"
    csvfile_path = "./data/image_name.csv"
    generateCSV(image_dir, mask_dir, csvfile_path)