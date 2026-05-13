#upload to gitee， 
import cv2
import numpy as np
import time
import math
from numpy import *
def generate_images_path():
    file = open('./datas/corel_dataset.txt', 'r', encoding="UTF-8")
    data1 = []
    for line in file:
        data_line = line.strip("\n").split()
        # print(data_line)
        data1.append(data_line)
    label_y = np.zeros((len(data1)))
    images_path = ['' for _ in range(len(data1))]
    for i in range(len(data1)):
        images_path[i] = data1[i][0]
        label_y[i] = data1[i][1]
    return images_path
def extract_cld_color_descriptor(image_path):
    image = cv2.imread(image_path)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # 计算CLD颜色描述符
    cld_descriptor = []
    for channel in range(3):
        channel_data = lab_image[:, :, channel]
        hist, _ = np.histogram(channel_data, bins=8, range=(0, 256), density=True)
        cld_descriptor.extend(hist)
    return cld_descriptor
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))
def retrieval_CLD(fea_one,all_fea,y_ac,y_test,topk):
    reeuc = np.zeros(len(all_fea))
    print(len(all_fea))
    i = 0
    for row in all_fea:
        dif = euclDistance(fea_one, row)
        reeuc[i] = dif
        i = i + 1
    sort_re = np.argsort(reeuc)
    sorted_arr = reeuc[sort_re]
    cou_ac = 0
    ap=0
    for i in range(0, topk):
        if y_test[int(sort_re[i])] == y_ac:
            cou_ac = cou_ac + 1
            tt = cou_ac / (i + 1)
            ap = ap + tt
    ap = ap / cou_ac
    return cou_ac,ap
if __name__ == '__main__':
    y_test = np.arange(10)
    y_test = np.repeat(y_test, 100)
    for i in range(1000):
        y_test[i] = y_test[i] + 1
    images_path=generate_images_path()
    cld_all=np.zeros((len(images_path),24))
    for i in range(len(images_path)):
        cld_all[i]=extract_cld_color_descriptor(images_path[i])
    topk=50
    cou_acc=0
    start_time = time.time()
    ap_acc = 0.0
    for i in range(0, 1000, 1):
        cou,ap = retrieval_CLD(cld_all[i], cld_all, y_test[i], y_test,topk)
        cou_acc = cou_acc + cou
        ap_acc = ap_acc + ap
    ap_acc = ap_acc / 1000
    end_time = time.time()
    elapsed_time = end_time - start_time
    cou=cou_acc/(1000*topk)