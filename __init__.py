import cv2
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

# imgPath = "data/train_images/Subset1_img__12.png"
# img_0 = cv2.imread(imgPath)
# mask = cv2.imread(imgPath.replace('images', 'labels'), 0)/255
# boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
# print(boxes.shape)
# label_0 = label(mask)
# props = regionprops(label_0)
# img_1 = img_0.copy()
# for prop in props:
#     print("found bounding box", prop.bbox)
#     cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
#
# mask_1 = np.expand_dims(mask, axis=0)
# pos = np.where(mask_1[0, :, :])
# xmin = np.min(pos[1])
# xmax = np.max(pos[1])
# ymin = np.min(pos[0])
# ymax = np.max(pos[0])
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.imshow(img_0)
# ax2.imshow(mask)
# ax3.imshow(img_1)
# plt.show()

imgPath = "data/train_images/Subset1_img__1.png"
img_0 = cv2.imread(imgPath)
mask = cv2.imread(imgPath.replace('images', 'labels'), 0)/255
label_0, count = ndi.label(mask, output=np.uint8)
print(np.unique(label_0), ", count:", count)
# masks = np.zeros((count, mask.shape[0], mask.shape[0]))
# labels = []
# for i in range(count):
#     masks[i, :, :] = np.where(label_0 == i+1, mask, 0)
#     labels.append(1)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.imshow(mask)
# ax2.imshow(masks[0, :, :])
# ax3.imshow(masks[1, :, :])
# plt.show()

a = np.expand_dims()