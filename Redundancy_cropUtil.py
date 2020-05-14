import glob
import os
from tqdm import tqdm
import argparse
import tqdm
import cv2 as cv
import numpy as np
from PIL import Image

def make_dir(new_dir):
    if  not os.path.exists(new_dir):
        os.makedirs(new_dir)

def subImg(img,i,j,targetSize, PaddingSize,height,width):
    if (i + 1) * targetSize < height and (j + 1) * targetSize < width:
        temp_img = img[targetSize * i: targetSize * i + targetSize+PaddingSize, targetSize * j: targetSize * j + targetSize+PaddingSize, :]
    elif (i + 1) * targetSize < height and (j + 1) * targetSize > width:
        temp_img = img[targetSize * i: targetSize * i + targetSize+PaddingSize, width - targetSize-PaddingSize: width, :]
    elif (i + 1) * targetSize > height and (j + 1) * targetSize < width:
        temp_img = img[height - targetSize-PaddingSize: height, targetSize * j: targetSize * j + targetSize+PaddingSize, :]
    else:
        temp_img = img[height - targetSize-PaddingSize: height, width - targetSize-PaddingSize: width, :]
    return temp_img


def crop(root, img_dir, label_dir,targetSize, PaddingSize, ImgSuffix, LabelSuffix):
    imgs_list  = glob.glob(root+"/images/*.tif")

    imgs_num = len(imgs_list)
    print("imgs_num:{}".format(imgs_num))
    make_dir(img_dir)
    make_dir(label_dir)

    for k in tqdm.tqdm(range(imgs_num)):
        # img = TIFF.open(imgs_list[k])
        # img = img.read_image()
        img = np.array(cv.imread(imgs_list[k]))
        imgName = os.path.split(imgs_list[k])[-1].split(".")[0]

        label = np.array(cv.imread(imgs_list[k].replace("images", "gt")))

        height ,width= img.shape[0], img.shape[1]
        rows , cols = height//targetSize+1, width//targetSize+1
        subImg_num = 0
        for i in range(rows):
            for j in range(cols):
                temp_img = subImg(img, i, j, targetSize, PaddingSize,height, width)
                temp_label = subImg(label, i, j, targetSize, PaddingSize, height, width)

                tempName = imgName + "_" + str(subImg_num) + ImgSuffix
                labelName = imgName + "_" + str(subImg_num) + LabelSuffix

                cv.imwrite(img_dir + '/' + tempName, temp_img)
                cv.imwrite(label_dir + '/' + labelName, temp_label)

                subImg_num +=1
                
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--root", type=str, default="./train",help='the path of input')
    parse.add_argument("--img_dir", type=str, default="./train_images",help='the path of images output')
    parse.add_argument("--label_dir", type=str, default="./train_labels",help='the path of labels output')
    parse.add_argument("--targetSize", type=int, default=230,help='the size of target')
    parse.add_argument("--PaddingSize", type=int, default=26,help='the size of padding')
    parse.add_argument("--LabelSuffix", type=str, default=".png",help='the suffix of label')
    parse.add_argument("--ImgSuffix", type=str, default=".png",help='the suffix of image')
    parse.add_argument("--isImg", type=bool, default=True,help='Img is true, np is false')

    args = parse.parse_args()
    crop(args.root, args.img_dir,args.label_dir, args.targetSize, args.PaddingSize, args.ImgSuffix,args.LabelSuffix)