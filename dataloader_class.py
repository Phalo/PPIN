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
import csv
import collections
import thop
from thop import profile
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
    def __init__(self, config, mode='train', transforms=None,fold=None, X=None, gts=None):
        """

        :param X: (BS,128*128)
        :param gts: (BS,N,2)
        """
        #self.__X = X
        self.__gts = gts
        self.__sigma = config['sigma']
        self.__debug_vis = config['debug_vis']

        self.__is_test = config['is_test']
        #fnames = glob.glob(config['path_image'] + "*.jpg")
        self.__fold = fold
        # gtnames =
        self.__gts = gts
        self.transforms = transforms

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
        self.num_landmark = 24


    def __len__(self):
        return len(self.__X)

    def __getitem__(self, item):
        H, W = 224,224
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



        #get_id
        if self.__fold is not None:
            x_name = self.__X[item]['image']
        else:
            x_name = self.__X[item]

        #read image


        #Img : pil.module modelu:RGB
        self.image_name = x_name.split('/')[5].split('.')[0]

        img, origin_size = self.readImage(
            os.path.join(self.path_Image, self.image_name+'.jpg')

        )


        #getkeypoints
        points = self.readLandmark(self.image_name,origin_size)
        # points : List[array]
        #H, W  =origin_size[0],origin_size[1]
        #print(H, W)
        # resize while keep ratio
        image_resize = image_aspect(img, H, W).change_aspect_rate().past_background().PIL2ndarray()
        # x: ndarray


        rate,offset = image_aspect(img, H, W).save_rate()
        gt_points = points * np.array([rate]) + offset


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


        if self.transforms is not None:
            #x = copy.deepcopy(x).astype(np.uint8).reshape(1,512, 512)
            #heatmaps = copy.deepcopy(heatmaps).astype(np.uint8).reshape(24, 512, 512)


            #image _equal

            # image_cv = cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
            # image_resize = cv2.equalizeHist(image_cv)
            image_transform, gt_points = self.transforms(image_resize, gt_points)


            # plt.imshow(image_transform)
            # plt.show()
            image_transform = image_transform.reshape(-1, H, W)
            image = np.asarray(image_transform)/1.0
            #img = np.repeat(image[:, :, np.newaxis], 3, axis=2).reshape(-1,H,W)
            image = torch.tensor(image,dtype=float)
            gt_points_relative = gt_points / H


            #image_transform = image_transform.reshape(H, W)
            #image = np.asarray(image_transform)/1.0
            #image = np.repeat(image[:, :, np.newaxis], 3, axis=2).reshape(-1,H,W)
            #image = torch.tensor(image,dtype=float)

            #
            #heatmaps = heatmaps.numpy().reshape(24,H,W)


        self.num_landmark = 24
        gt = torch.zeros((self.num_landmark, H, W), dtype=torch.float)
        mask = torch.zeros((self.num_landmark, H, W), dtype=torch.float)
        guassian_mask = torch.zeros((self.num_landmark, H, W), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, H, W), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, H, W), dtype=torch.float)

        # y, x = image_resize.shape[0], image_resize.shape[1]
        #
        # for i, landmark in enumerate(gt_points):
        #     if int(landmark[1])==512 or int(landmark[0])==512:
        #         gt[i][int(landmark[1])-1][int(landmark[0])-1] = 1
        #     else:
        #         gt[i][int(landmark[1])][int(landmark[0]) ] = 1
        #     margin_x_left = int(max(0, landmark[0] - self.Radius))
        #     margin_x_right = int(min(x, landmark[0] + self.Radius))
        #     margin_y_bottom = int(max(0, landmark[1] - self.Radius))
        #     margin_y_top = int(min(y, landmark[1] + self.Radius))
        #
        #     mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
        #         self.mask[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]
        #     #print(torch.max(mask[i]))
        #     guassian_mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
        #         self.guassian_mask[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]
        #     offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
        #         self.offset_x[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]
        #     offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
        #         self.offset_y[0:margin_y_top - margin_y_bottom, 0:margin_x_right - margin_x_left]


        #label = torch.as_tensor(label)
        #print(label)
        #x = np.array(x).reshape((1, H, W)).astype(np.float32)
        #heatmaps = heatmaps.astype(np.float32)
        info = {
            "image_id": self.image_name,
            "image_width": origin_size[0],
            "image_height": origin_size[1],
            "obj_origin_hw": [H, W],
            "key" : gt_points,
            "keypoints": gt_points_relative,
            "loss_mask": loss_mask,
            "label": labels,
            "groundtruth":gt,
            #"heatmaps" :mask,
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
        points_list=[]
        label_list = []
        for point in points:
            points_list.append(point.tolist())
        for label in labels:
            label_list.append(label.tolist())

        # #c
        # #headers = ('id', 'origin_size', 'scale', 'pad', 'landmark')
        keypoints = {'id': self.image_name,'origin_size':(origin_size[0],origin_size[1]),'scale':rate,
                     'pad':[offset[0],offset[1]],'landmark':points_list,'labels':label_list}



        return image, info


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



from PIL import Image, ImageDraw,ImageFilter
import os

if __name__ == '__main__':
    #from train import config
    config = dict()
    config['lr'] = 0.01
    config['momentum'] = 0.009
    config['weight_decay'] = 1e-4
    config['epoch_num'] = 100
    config['batch_size'] = 2
    config['sigma'] = 2.5
    config['debug_vis'] = False

    config['train_fname'] = ''
    config['test_fname'] = ''
    # config ['path_image'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'
    config['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_test/'
    config['train_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'

    config['path_label'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/'
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
    data_transform = {
        "train": transform_new.Compose([
                                     #transform_new.RandomCrop(),
                                     #transform_new.Resize(512,512),
                                     #transforms.ReservePixel(),
                                     transform_new.RandomHorizontalFlip(0),
                                     #transform_new.ToTensor()
        ]),
        "val": transform_new.Compose([transform_new.RandomHorizontalFlip(0),
                                   #transforms.Resize(512,512)
        ])
    }
    dataset = KFDataset(config, 'test',transforms=data_transform["val"])
    #dataset = KFDataset(config, mode='train', transforms=None)

    dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    label_non =[]
    label_non_rate = []
    keypoints_all = []
    names = []
    count = 0

    # ## check dataloader
    # for i, (x, info) in enumerate(dataset):
    #
    #     #keypoints_all.append(keypoints)
    #
    #     #print(len(info["keypoints"]))
    #     #label_non.append(sum(j>0 for j in info["label"])/24)
    #     #non  = sum(j>0 for j in info["label"][i%1])
    #     #print('batch')
    #     for key_p in info["keypoints"]:
    #         #print(key_p)
    #         if key_p[0]==0 or key_p[1]==0:
    #
    #             count = count +1
    #             names.append(info["image_id"])
    # #print("There are {} data with 0".format(count))
    #
    # for name in names:
    #     print(name)

    name_recheck =[
        'a8abeb180af0036938f087c3bb99f279',
        '4e382181f40896ed6b12a6ff7b44ab16',
        '3bc974ed58538d58c01595e69b09a92a',
        '93ee7824ef6d3b0b304bd23ec9d37c7f',
        '23d8dd0f712204b85da1b3bf8d577ffa',
        '3c3be8f9818739957e621e777372cd86',
        'b6720313c03637f23bcee8f9cdfa04ac',
        '1589d99a7b3210aecaf64d0f988f6547',
        'ebd604684f5b3106fcf3f50445a8d641',
        '2af467009ba7ed4c709115a92bdc28e7',
        '9025055e7a039288c65b8eed5e8a4209',
        '60cf5394893e0f110b5743818b5c86e4',
        '929ee4a5b953f032a6d77c16deb52d71',
        '7d28b5592f5b7ca09ed8492af798139a',
        '9c5060a5fd953cbd3c65d649aaef9cea',
        'e711b85e8c74d073b5ffa1b352d164bb',
        'f59bd84ab10341434cfffd06326b6518',
        '54352f879934e18f06df0d2c80a9bdd8',
        '132b13399a67c7cf1e01358e946a668b',
        'b60d083b8f7e791879dfeb6f0abc8554',
        'df11d8b51c45fdb565eb3618c3d2a96b',
        '47905626121b99fa37175b60bd483fd5',
        '845198d0cbebee9a54b6a298d66dd523',
        '1e936b326bd3f6d34e7e079b63654d2a',
        '47e53467ce62f1e5304a6a1531082f49',
        'c51372ddaca17880f03ada93e0d9a5aa',
        '20ce6e8bd52eb238e593c1f6adbb8609',
        '8f971b60fd2dd0a5c802dd119585df20',
        'c7c499caa00c266280f89fa48e9e1141',
        '574fb0083eb2a1e80684422323c3872a',
        '9f20b4ccffc865c50d3cd7a22a19b333',
        '2c8886c3a8bc3015198a361b0750f166',
        '248ab967eba2f1df3cac806b429dc08e',
        '65a20b68cb3ab6042d3b734ecc3838f8',
        '614b3b521407e115a83193e814be8142',
        '74370c65143cf3d8c70991a6d2d2e38e',
        '51a0fe0f31164a4e507754c689891305',
        '65f140000f9cc691dcdec64bef6ab5a3',
        '162e61e1204c12d8795b546b8592abcb',
        'd088471c7445ce4585d7a50030b5f7bb',
        '1c7ea15e463ba34cfd2f5dc75d9e091b',
        '353ee8068f08f99a7e0b98c0862f0472',
        'fe521258ff5b2fc1e6151d89db8d8a37',
        '02b19951218b1bc17597b7dc6bf13e70',
        'a1949421345d967feaa903d7a7816ee6',
        'de7e0eb6d52aeb9c074e6fce18914908',
        '004004095d8a302b1c0815ccb044c018',
        '2ccd1f744f0945b2967efc98990d4c34',
        'c0e608fc717f0811052d0357a10abe8c',
        '2bd198cf363f45391d955b73028c0b9a',
        'c728733c147e60e283850c9df565da4b',
        'afc5a9e760859e8ddd281cd532eb54a4',
        'c58aa2c79109b51c7b8459cd772013f2',
        'dff336da37e5204d82c10a61ba15fe16',
        '25c0a7c3eb5fc5b75f28f138ad0c4786',

    ]
    data_root = r"/public/huangjunzhang/KeyPointsDetection-master/mask_test2/"
    npy_root = r"/public/huangjunzhang/KeyPointsDetection-master/mask_npy_test2/"
    mask_check =r"/public/huangjunzhang/KeyPointsDetection-master/recheck/"
    for i, (x, info) in enumerate(dataset):
        if  info["image_id"] in name_recheck:


        # create mask

            print("checking mask")
            height,width = 512,512
            mask = Image.new("L", (width, height), 0)
            #create_mask
            # 假设关键点坐标为 (x1, y1), (x2, y2), (x3, y3), (x4, y4)
            keypoints = info["keypoints"].reshape(6,4,2)
            for i in range(6):
                #print(keypoints[i,:,:][0,:][1])
                key_ = np.array([keypoints[i,:,:][0,:][0],keypoints[i,:,:][0,:][1]])
                key_2 = np.array([keypoints[i, :, :][1, :][0], keypoints[i, :, :][1, :][1]])
                key_3 = np.array([keypoints[i, :, :][2, :][0], keypoints[i, :, :][2, :][1]])
                key_4 = np.array([keypoints[i, :, :][3, :][0], keypoints[i, :, :][3, :][1]])
                key_mask = [(key_[0],key_[1]),(key_2[0],key_2[1]),(key_4[0],key_4[1]),(key_3[0],key_3[1])]
                # 使用ImageDraw绘制多边形
                draw = ImageDraw.Draw(mask)
                draw.polygon(key_mask, outline=i+1,fill=i+1)  # 绘制多边形区域

            # 可选：对掩码进行模糊处理
            mask = mask.filter(ImageFilter.GaussianBlur(radius=2))

            # plt.imshow(mask)
            # plt.show()
            # 保存掩码为图像文件
            output_path = os.path.join(data_root,"mask_{}.png".format(info["image_id"]))
            output_check = os.path.join(mask_check, "mask_{}.png".format(info["image_id"]))
            out_npy_path =os.path.join(npy_root,"{}.npy".format(info["image_id"]))
            mask_np = np.array(mask)
            print(mask_np.max())
            if mask_np.max()!=6:
                print("Warning: in '{}' mask, there are some wrong <=0".format(info["image_id"]))
            plt.imshow(mask_np)
            plt.savefig(output_check)
            np.save(out_npy_path,mask_np)
            mask.save(output_path)
            print(f"Mask saved to {output_path}")
        else :
            print("skip data")


        ####check _ 20230827
        # if  info["image_id"] in names:
        #     print("skip data")
        #
        # # create mask
        # else:
        #     print("creating mask")
        #     height,width = 512,512
        #     mask = Image.new("L", (width, height), 0)
        #     #create_mask
        #     # 假设关键点坐标为 (x1, y1), (x2, y2), (x3, y3), (x4, y4)
        #     keypoints = info["keypoints"].reshape(6,4,2)
        #     for i in range(6):
        #         #print(keypoints[i,:,:][0,:][1])
        #         key_ = np.array([keypoints[i,:,:][0,:][0],keypoints[i,:,:][0,:][1]])
        #         key_2 = np.array([keypoints[i, :, :][1, :][0], keypoints[i, :, :][1, :][1]])
        #         key_3 = np.array([keypoints[i, :, :][2, :][0], keypoints[i, :, :][2, :][1]])
        #         key_4 = np.array([keypoints[i, :, :][3, :][0], keypoints[i, :, :][3, :][1]])
        #         key_mask = [(key_[0],key_[1]),(key_2[0],key_2[1]),(key_4[0],key_4[1]),(key_3[0],key_3[1])]
        #         # 使用ImageDraw绘制多边形
        #         draw = ImageDraw.Draw(mask)
        #         draw.polygon(key_mask, outline=i+1,fill=i+1)  # 绘制多边形区域
        #
        #     # 可选：对掩码进行模糊处理
        #     mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
        #
        #     # plt.imshow(mask)
        #     # plt.show()
        #     # 保存掩码为图像文件
        #     output_path = os.path.join(data_root,"mask_{}.png".format(info["image_id"]))
        #     output_check = os.path.join(mask_check, "mask_{}.png".format(info["image_id"]))
        #     out_npy_path =os.path.join(npy_root,"{}.npy".format(info["image_id"]))
        #     mask_np = np.array(mask)
        #
        #     if mask_np.max()!=6:
        #         print("Warning: in '{}' mask, there are some wrong <=0".format(info["image_id"]))
        #     plt.imshow(mask_np)
        #     plt.savefig(output_check)
            #plt.show()

            # print(mask_np.shape)
            # np.save(out_npy_path,mask_np)
            # mask.save(output_path)
            # print(f"Mask saved to {output_path}")

    #print("There are {} data with 0".format(count))



    #save keypoint
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



