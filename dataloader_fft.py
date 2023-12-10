# coding=utf-8

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy
import glob
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
from resize import image_aspect
import os
#import scratch_3
from typing import Tuple
from lxml import etree
import math
import transform_new
#from utils_freq.freq_pixel_loss import find_fake_freq, get_gaussian_kernel
from get_fft import find_fake_freq, get_gaussian_kernel
import csv
import collections
#from train import config
index = [
    'C2_TR',
    'C2_TL',
    'C2_DR',
    'C2_DL',
    'C3_TR',
    'C3_TL',
    'C3_DR',
    'C3_DL',
    'C4_TR',
    'C4_TL',
    'C4_DR',
    'C4_DL',
    'C5_TR',
    'C5_TL',
    'C5_DR',
    'C5_DL',
    'C6_TR',
    'C6_TL',
    'C6_DR',
    'C6_DL',
    'C7_TR',
    'C7_TL',
    'C7_DR',
    'C7_DL']

def plot_sample(x, y, axis):
    """

    :param x: (9216,)
    :param y: (4,2)
    :param axis:
    :return:
    """
    img = x.reshape(128, 128)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[:, 0], y[:, 1], marker='x', s=10)


def plot_demo(X, y):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y[i], ax)

    plt.show()

def scale_box(xmin: float, ymin: float, w: float, h: float, scale_ratio: Tuple[float, float]):
    """根据传入的h、w缩放因子scale_ratio，重新计算xmin，ymin，w，h"""
    s_h = h * scale_ratio[0]
    s_w = w * scale_ratio[1]
    xmin = xmin - (s_w - w) / 2.
    ymin = ymin - (s_h - h) / 2.
    return xmin, ymin, s_w, s_h



class KFDataset(Dataset):
    def __init__(self, config, mode='train', transforms=None, fold=None, X=None, gts=None,lumbar=False):
        """

        :param X: (BS,128*128)
        :param gts: (BS,N,2)
        """

        #self.fold = fold
        self.__gts = gts
        self.__sigma = config['sigma']
        self.__debug_vis = config['debug_vis']

        self.__is_test = config['is_test']
        #fnames = glob.glob(config['path_image'] + "*.jpg")

        # gtnames =
        self.__gts = gts
        self.transforms = transforms
        self.__fold = fold

        # if self.__fold is not None:
        #     #print("five fold evaluation, using",self.__fold)
        #     fnames = self.__fold
        #     self.path_Image = config['train_image_path']
        # else:
        #     fnames = glob.glob(self.path_Image + "*.jpg")
        #     if mode =='train':
        #         self.path_Image = config['train_image_path']
        #     else:
        #         self.path_Image = config['test_image_path']
        # self.__X = fnames
        # self.path_label = config['path_label']
        # self.num_landmark = 24


        #self.__heatmap = config['heatmap_path']
        if mode =='train':
            self.path_Image = config['train_image_path']
        else:
            self.path_Image = config['test_image_path']

        if self.__fold is not None:
            print("five fold evaluation")
            #using five fold validation
            self.path_Image = config['train_image_path']
            fnames = self.__fold
        else:
            fnames = glob.glob(self.path_Image + "*.jpg")

        self.__X = fnames
        self.path_label = config['path_label']
        self.lumbar = lumbar
        if lumbar:

            print("Creating dataloader for lumbar training")
            self.num_landmark = 20
        else:
            print("Creating dataloader for cervical training")
            self.num_landmark = 24




    def __len__(self):
        return len(self.__X)

    def __getitem__(self, item):
        H, W = 512,512
        size = [512,512]
        R_ratio = 0.02
        self.Radius = int(max(size)* R_ratio)
        # gen mask
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1
                    # for guassian mask
                    guassian_mask[i][j] = math.exp(-0.5 * math.pow(distance, 2) /\
                        math.pow(self.Radius, 2))
        self.mask = mask
        #print(torch.max(mask))
        self.guassian_mask = guassian_mask


        # gen offset
        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius



        #load image
        if self.__fold is not None:
            x_name = self.__X[item]['image']
        else:
            x_name = self.__X[item]



        #x_name_fold = self.__fold[item]


        #get_id
        self.image_name = x_name.split('/')[5].split('.')[0]
        #print("processing" ,self.image_name)
        #read image
        img, origin_size = self.readImage(
            os.path.join(self.path_Image, self.image_name+'.jpg')

        )

        #Img : pil.module modelu:RGB

        #getkeypoints
        points = self.readLandmark(self.image_name, origin_size)
        # points : List[array]

        # resize while keep ratio
        image_resize = image_aspect(img, H, W).change_aspect_rate().past_background().PIL2ndarray()
        # x: ndarray


        rate,offset = image_aspect(img, H, W).save_rate()
        gt_points = points * np.array([rate]) + offset
        #print(gt_points)

        #create loss_mask
        gt_weight = np.ones((len(points),),dtype=np.float32)
        for i in range(len([points])):
            if points[i][1]==0 :
                gt_weight[i]= 0
        loss_mask = torch.as_tensor(gt_weight, dtype=torch.float32)

        #loss_mask :Tensor


        # loading heatmaps from .npy file
        #heatmaps = self.load_heatmap(self.path_label,self.image_name)

        bboxs = self.readbbox(self.image_name)
        labels = self.create_label(points, bboxs)

        gauss_kernel = get_gaussian_kernel(size=21)

        if self.transforms is not None:
            #x = copy.deepcopy(x).astype(np.uint8).reshape(1,512, 512)
            #heatmaps = copy.deepcopy(heatmaps).astype(np.uint8).reshape(24, 512, 512)


            #image _equal

            # image_cv = cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
            # image_resize = cv2.equalizeHist(image_cv)
            image_transform, gt_points = self.transforms(image_resize, gt_points)
            image_transform = image_transform.reshape(-1, H, W)
            image = np.asarray(image_transform)/1.0
            #img = np.repeat(image[:, :, np.newaxis], 3, axis=2).reshape(-1,H,W)
            image = torch.tensor(image,dtype=float)
            real_img_freq = find_fake_freq(image, gauss_kernel)
            image_dual  =real_img_freq[0]

            #create fft image
            ### check image
            imgae_hf = image_dual[0,:,:]
            imgae_lf = image_dual[1, :, :]

            ##ploty
            # plt.figure(0)
            # plt.subplot(1,2,1)
            # plt.imshow(imgae_hf,cmap='gray')
            # plt.subplot(1,2,2)
            # plt.imshow(imgae_lf,cmap='gray')
            # plt.show()



        #self.num_landmark = 20
        gt = torch.zeros((self.num_landmark, H, W), dtype=torch.float)
        mask = torch.zeros((self.num_landmark, H, W), dtype=torch.float)
        guassian_mask = torch.zeros((self.num_landmark, H, W), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, H, W), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, H, W), dtype=torch.float)

        y, x = image_resize.shape[0], image_resize.shape[1]

        for i, landmark in enumerate(gt_points):
            if int(landmark[1])==H or int(landmark[0])==W:
                gt[i][int(landmark[1])-1][int(landmark[0])-1] = 1
            else:
                gt[i][int(landmark[1])][int(landmark[0]) ] = 1
            margin_x_left = int(max(0, landmark[0] - self.Radius))
            margin_x_right = int(min(x, landmark[0] + self.Radius))
            margin_y_bottom = int(max(0, landmark[1] - self.Radius))
            margin_y_top = int(min(y, landmark[1] + self.Radius))

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]
            #print(torch.max(mask[i]))
            guassian_mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.guassian_mask[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_x[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_y[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]

        #label = torch.as_tensor(label)
        #print(label)
        #x = np.array(x).reshape((1, H, W)).astype(np.float32)
        #heatmaps = heatmaps.astype(np.float32)
        info = {
            "image_id": self.image_name,
            "image_width": origin_size[0],
            "image_height": origin_size[1],
            "obj_origin_hw": [H, W],
            "keypoints": gt_points,
            "loss_mask": loss_mask,
            "label": labels,
            "groundtruth":gt,
            "heatmaps" :mask,
            "offset_x" :offset_x,
            "offset_y" :offset_y,

        }

            #gt = gt.numpy().reshape(24,2)

        if self.__debug_vis == True:
            #for i in range(heatmaps.shape[0]):
                #x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                #img = copy.deepcopy(x).astype(np.uint8).reshape(H,W,1)
            #print(torch.max(mask))
                #self.visualize_heatmap_target(image_transform ,points, bboxs, copy.deepcopy(heatmaps), self.image_name)\
            self.visualize_heatmap_target(image_transform, gt_points, bboxs, mask, self.image_name)


        #print(label)
        # points_list=[]
        # label_list = []
        # for point in points:
        #     points_list.append(point.tolist())
        # for label in labels:
        #     label_list.append(label.tolist())
        #
        # # #c
        # # #headers = ('id', 'origin_size', 'scale', 'pad', 'landmark')
        # keypoints = {'id': self.image_name,'origin_size':(origin_size[0],origin_size[1]),'scale':rate,
        #              'pad':[offset[0],offset[1]],'landmark':points_list,'labels':label_list}



        return image_dual,info,image_resize


    #



    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def readbbox(self, name):
        boxes = []
        root_dir = "/public/huangjunzhang/KeyPointsDetection-master/Annotations/"
        xml_path = root_dir+name+".xml"
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        for obj in data["object"]:

            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
        return boxes


    def create_label(self,points ,bboxs):
        label = np.zeros((self.num_landmark))
        for box in bboxs:
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            for i in range(len(label)):
                 if xmin<points[i][0]< xmax and ymin<points[i][1]<ymax :
                     label[i] = 1
        return label


    def CenterGaussianHeatMap(self,keypoints, height, weight, variance):

        c_x = keypoints[0]
        c_y = keypoints[1]
        gaussian_map = np.zeros((height, weight))
        for x_p in range(weight):
            for y_p in range(height):
                dist_sq = (x_p - c_x) * (x_p - c_x) + \
                          (y_p - c_y) * (y_p - c_y)
                exponent = dist_sq / 2.0 / variance / variance
                gaussian_map[y_p, x_p] = np.exp(-exponent)
        # normalize
        xmax = max(map(max, gaussian_map))
        xmin = min(map(min, gaussian_map))
        gaussian_map_nor = (gaussian_map-xmin)/(xmax-xmin)
        #Gau = Image.fromarray(gaussian_map)
        #Gau.show()
        return gaussian_map_nor

    def _putGaussianMaps(self,keypoints, crop_size_y, crop_size_x, sigma):
        """

        :param keypoints: (24,2)
        :param crop_size_y: int  512
        :param crop_size_x: int  512
        :param stride: int  1
        :param sigma: float   1e-
        :return:
        """
        all_keypoints = keypoints #4,2
        point_num = len(all_keypoints)  # 4
        heatmaps_this_img = []
        for k in range(point_num):  # 0,1,2,3
            #flag = ~np.isnan(all_keypoints[k,0])
            #heatmap = self._putGaussianMap(all_keypoints[k],flag,crop_size_y,crop_size_x,stride,sigma)
            heatmap = self.CenterGaussianHeatMap(keypoints=all_keypoints[k], height=crop_size_y, weight=crop_size_x, variance=sigma)
            heatmap = heatmap[np.newaxis,...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img,axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
        np.save('./crop/{}'.format(self.image_name),heatmaps_this_img)
        print('save done')
        return heatmaps_this_img

    def load_heatmap(self, path, image_name):
        npy_path = str(path) + str(image_name)
        heatmaps_this_img = np.load("{}.npy".format(npy_path))
        return heatmaps_this_img


    def readImage(self,path):
        img = Image.open(path).convert('RGB')
        origin_size = img.size
        return img,origin_size

    def readLandmark(self, name, origin_size):

        path = os.path.join(self.path_label, name+'_jpg_Label.json')
        kp = []

        with open (path, 'r') as f:
            gt_json = json.load(f)
            #get label
            mark_list_model = gt_json['Models']['LandMarkListModel']
            points = mark_list_model['Points'][0]['LabelList']


            for i in range(self.num_landmark):
                if i >=len(points):
                    landmark = np.array([0,0])
                    kp.append((i+2+len(points),landmark))

                else:
                    landmark = np.array([points[i]['Position'][0],points[i]['Position'][1]])
                    kp.append((points[i]['Label'],landmark))
                #get landmark
            kp.sort(reverse=False)
            points_in_image = []
            for j in range(self.num_landmark):
                points_in_image.append(kp[j][1])

            #print('end')




            # for i in range(self.num_landmark):
            #     landmark= [float(i) for i in f.readline().split(',')]
            #     points.append(landmark)
            # points = np.array(points)

        return points_in_image

    # def draw_box(image, boxes, classes, keypoints, scores, category_index, thresh=0.5, line_thickness=8):
    #     box_to_display_str_map = collections.defaultdict(list)
    #     box_to_color_map = collections.defaultdict(str)
    #
    #     filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map)
    #
    #     # Draw all boxes onto image.
    #     draw = ImageDraw.Draw(image)
    #     im_width, im_height = image.size
    #     for box, color in box_to_color_map.items():
    #         xmin, ymin, xmax, ymax = box
    #         (left, right, top, bottom) = (xmin * 1, xmax * 1,
    #                                       ymin * 1, ymax * 1)
    #         draw.line([(left, top), (left, bottom), (right, bottom),
    #                    (right, top), (left, top)], width=line_thickness, fill=color)
    #         draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)
    #     for x, y in keypoints:
    #         shape = [(x - 5, y - 5), (x + 5, y + 5)]
    #         draw.ellipse(shape, fill="#ffff33")



    def visualize_heatmap_target(self, oriImg, gt, bbox, heatmap,name):

        oriImg = oriImg.reshape(512,512)
        stacked_img = np.stack((oriImg,) * 3, axis=-1).reshape(512,512,3)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        plt.subplot(2,2,1)
        plt.imshow(oriImg.astype(np.uint8),cmap=plt.get_cmap('gray'))
        plt.subplot(2,2,2)
        plt.imshow(stacked_img,cmap=plt.get_cmap('gray'))

        # for j in range(len(bbox)):
        #     x = bbox[j][0]
        #     y = bbox[j][1]
        #     width = bbox[j][2]-bbox[j][0]
        #     height = bbox[j][3]-bbox[j][1]
        #     rect = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')
        #     ax.add_patch(rect)
        for i in range(len(gt)):
            plt.scatter(gt[i][0], gt[i][1])
            plt.text(gt[i][0], gt[i][1], '{}'.format(index[i]), color='g')
        #plt.savefig('./Input_train/{}.jpg'.format(name))
        #plt.show(block=False)
        #plt.pause(2)
        #plt.close()

        plt.figure(2)
        for i in range(24):
            plt.subplot(4, 6, i+1)
            #plt.imshow(oriImg)
            plt.imshow(heatmap[i],cmap=plt.get_cmap('gray'))

        plt.show()





if __name__ == '__main__':
    #from train import config
    from sklearn.model_selection import KFold
    config = dict()
    config['lr'] = 0.01
    config['momentum'] = 0.009
    config['weight_decay'] = 1e-4
    config['epoch_num'] = 100
    config['batch_size'] = 2
    config['sigma'] = 2.5
    config['debug_vis'] = True

    config['train_fname'] = ''
    config['test_fname'] = ''
    # config ['path_image'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'
    config['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_test/'
    config['train_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'

    config['path_label'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/lumbar_json/'
    config['path_label_train'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/train_json/'
    # config['json_path']='/public/huangjunzhang/test/keypoints_train.json'
    config['is_test'] = False

    config['save_freq'] = 10
    config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints/kd_MLT_epoch_499_model.ckpt'
    config['start_epoch'] = 0
    config['load_pretrained_weights'] = False
    config['eval_freq'] = 50
    config['debug'] = False
    config['featurename2id'] = {
        'C2_TR': 0,
        'C2_TL': 1,
        'C2_DR': 2,
        'C2_DL': 3,
        'C3_TR': 4,
        'C3_TL': 5,
        'C3_DR': 6,
        'C3_DL': 7,
        'C4_TR': 8,
        'C4_TL': 9,
        'C4_DR': 10,
        'C4_DL': 11,
        'C5_TR': 12,
        'C5_TL': 13,
        'C5_DR': 14,
        'C5_DL': 15,
        'C6_TR': 16,
        'C6_TL': 17,
        'C6_DR': 18,
        'C6_DL': 19,
        'C7_TR': 20,
        'C7_TL': 21,
        'C7_DR': 22,
        'C7_DL': 23,
    }
    images1 = sorted(glob.glob(os.path.join(config['train_image_path'], '*.jpg')))
    labels1 = sorted(glob.glob(os.path.join(config['path_label_train'], '*_jpg_Label.json')))
    floder = KFold(n_splits=5, random_state=42, shuffle=True)
    data_dicts1 = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(images1, labels1)]
    #
    train_files = []
    test_files = []
    for k, (Trindex, Tsindex) in enumerate(floder.split(data_dicts1)):
        train_files.append(np.array(data_dicts1)[Trindex].tolist())
        test_files.append(np.array(data_dicts1)[Tsindex].tolist())
    data_transform = {
        "train": transform_new.Compose([
                                     #transform_new.RandomCrop(),
                                     #transform_new.Resize(512,512),
                                     #transforms.ReservePixel(),
                                     transform_new.RandomHorizontalFlip(0.5),
                                     transform_new.ToTensor()]),
        "val": transform_new.Compose([transform_new.ToTensor(),
                                   #transforms.Resize(512,512)
        ])
    }
    dataset = KFDataset(config, 'train',transforms=data_transform["train"],fold=train_files[0])
    #dataset = KFDataset(config, mode='train', transforms=None)

    dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    # label_non =[]
    # label_non_rate = []
    # keypoints_all = []
    for i, (x, info) in enumerate(dataLoader):

        #keypoints_all.append(keypoints)

        print(info["label"])
        # #label_non.append(sum(j>0 for j in info["label"])/24)
        #
        # non  = sum(j>0 for j in info["label"][i%1])
        # label_non.append(int(non))


        print('batch')

    # headers = ('id', 'origin_size', 'scale', 'pad', 'landmark','labels')
    # with open('test_vert2.0.csv', 'w', encoding='utf-8', newline='')as f:
    #     write = csv.DictWriter(f,headers)
    #     write.writeheader()
    #
    #
    #     for keypoints in keypoints_all:
    #          #write.writerow(keypoints)
    #          write.writerow(keypoints)


    # print(label_non)
    # for non in label_non:
    #     non /= 24
    #     label_non_rate.append(non)
    # label = np.array(label_non)
    # rate = np.array(label_non_rate)
    # plt.plot()
    # for i in range(len(rate)):
    #     plt.scatter(i,rate[i],s=5,c='b')
    # plt.show()
