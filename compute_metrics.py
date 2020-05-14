import numpy as np
import glob
import tqdm
from PIL import Image
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from skimage import io
from skimage import measure
from scipy import ndimage
from sklearn.metrics import f1_score


def get_buildings(mask, pixel_threshold):
    gt_labeled_array, gt_num = ndimage.label(mask)
    unique, counts = np.unique(gt_labeled_array, return_counts=True)
    for (k, v) in dict(zip(unique, counts)).items():
        if v < pixel_threshold:
            mask[gt_labeled_array == k] = 0
    return measure.label(mask, return_num=True)


def calculate_f1_buildings_score(y_pred_path, iou_threshold=0.45, component_size_threshold=100):
    tp = 0
    fp = 0
    fn = 0

    y_pred_list = glob.glob(f"./{y_pred_path}/*.png")

    for m in tqdm.tqdm(range(len(y_pred_list))):
        processed_gt = set()
        matched = set()

        mask_img = cv.imread(y_pred_list[m], 0)/255
        gt_mask_img = cv.imread(y_pred_list[m].replace(f"{y_pred_path}","validLabel").replace("RGB", "GTC"), 0)/255

        predicted_labels, predicted_count = get_buildings(mask_img, component_size_threshold)
        gt_labels, gt_count = get_buildings(gt_mask_img, component_size_threshold)

        gt_buildings = [rp.coords for rp in measure.regionprops(gt_labels)]
        pred_buildings = [rp.coords for rp in measure.regionprops(predicted_labels)]
        gt_buildings = [to_point_set(b) for b in gt_buildings]
        pred_buildings = [to_point_set(b) for b in pred_buildings]
        for j in range(predicted_count):
            match_found = False
            for i in range(gt_count):
                pred_ind = j + 1
                gt_ind = i + 1
                if match_found:
                    break
                if gt_ind in processed_gt:
                    continue
                pred_building = pred_buildings[j]
                gt_building = gt_buildings[i]
                intersection = len(pred_building.intersection(gt_building))
                union = len(pred_building) + len(gt_building) - intersection
                iou = intersection / union
                if iou > iou_threshold:
                    processed_gt.add(gt_ind)
                    matched.add(pred_ind)
                    match_found = True
                    tp += 1
            if not match_found:
                fp += 1
        fn += gt_count - len(processed_gt)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f_score = 2 * precision * recall / (precision + recall)
    return f_score


def to_point_set(building):
    return set([(row[0], row[1]) for row in building])


#计算平均数
def mean(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def mean_iou(input, target, classes = 2):
    # input = input[:target.shape[0],:target.shape[1]]
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return  miou/2


def iou(input, target, classes = 1):
    intersection = np.logical_and(target == classes, input == classes)
    # print(intersection.any())
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return  iou


y_pred_path = "predict_UNet_55"
imglist = glob.glob(f"./{y_pred_path}/*.png")
# imglist = glob.glob("./post_processing_labels/*.png")

num = len(imglist)
# metric = []
MIOU = 0.0
max = 0
min = 1
count1=0
count2=0
metric = []
#########compute iou
for i in tqdm.tqdm(range(num)):
    name = os.path.split(imglist[i])[-1].split(".")[0][0:-3]+"GTC"+".png"
    targetPath = "./validLabel/"+name
    img = np.array(cv.imread(imglist[i], 0))/255
    target = np.array(cv.imread(targetPath, 0))/255
    iou_score = iou(img, target)
    metric.append(iou_score)

print(np.mean(metric))

#########compute f1
metric = []
imgs = []
targets = []
for i in tqdm.tqdm(range(num)):
    name = os.path.split(imglist[i])[-1]
    targetPath = "./validLabel/"+name.replace("RGB","GTC")
    img = np.array(io.imread(imglist[i]))/255
    target = np.array(io.imread(targetPath))/255
    imgs.append(img.tolist()), targets.append(target.tolist())
    # matrix = confusion_matrix(y_true=target, y_pred=img)
    # metric.append(matrix)
    # matrix2 +=matrix.astype(np.uint64)
imgs, targets =np.array(imgs).flatten(), np.array(targets).flatten()
f1 = f1_score(y_true=targets, y_pred=imgs)
print("f1:{}".format(f1))

##########instance f1
instance_f1_score = calculate_f1_buildings_score(y_pred_path, iou_threshold=0.45, component_size_threshold=100)
print(f"{y_pred_path}:{instance_f1_score}")
