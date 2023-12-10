# coding=utf-8

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import models
import seaborn as sns
from data_loader import KFDataset
from U2Netnew import U2Net
from MMWUet import MMWUNet
#from train_single_gpu import config, get_peak_points, get_mse
from train import  config,get_mse,get_peak_points
from collections.abc import Iterable
from sklearn import metrics


#THRESHOLD = [2, 2.5, 3, 4, 6, 9, 10]
THRESHOLD =[3,6,9,10]
DRAW_TEXT_SIZE_FACTOR = { 'cephalometric': 1.13, 'hand': 1, 'chest': 1.39}


def np2py(obj):
    if isinstance(obj, Iterable):
        return [np2py(i) for i in obj]
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj
 #计算距离
def radial(pt1, pt2 ,factor=1):
    if  not isinstance(factor,Iterable):
        factor = [factor]*len(pt1)
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5



def get_sdr(distance_list, threshold =THRESHOLD):
    ret = {}
    n = len(distance_list)
    for th in threshold:
        ret[th]=sum(d<= th for d in distance_list)/n
    return ret

def cal_all_distance(points, gt_points, factor ):
    n1 = len(points)
    n2 = len(gt_points)
    factor_index = np.array(factor) * 1
    if n1 == 0:
        print("[Warning]: Empty input for calculating mean and std")
        return 0, 0
    if n1 != n2:
        raise Exception("Error: lengthes dismatch, {}<>{}".format(n1, n2))
    return [radial(p, q, factor) for p, q ,factor in zip(points, gt_points,factor_index)]

def analysis_all(li1):
    summary = {}
    mean1, std1, = np.mean(li1), np.std(li1)
    sdr1 = get_sdr(li1)
    n = len(li1)
    summary['LANDMARK_NUM'] = n
    summary['MRE'] = np2py(mean1)
    summary['STD'] = np2py(std1)
    summary['SDR'] = {k: np2py(v) for k, v in sdr1.items()}
    print('MRE:', mean1)
    print('STD:', std1)
    print('SDR:')
    for k in sorted(sdr1.keys()):
        print('     {}: {}'.format(k, sdr1[k]))
    return summary
def analysis(li1):
    summary = {}
    mean1, std1, = np.mean(li1), np.std(li1)
    sdr1 = get_sdr(li1)
    n = len(li1)
    summary['LANDMARK_NUM'] = n
    summary['MRE'] = np2py(mean1)
    summary['STD'] = np2py(std1)
    summary['SDR'] = {k: np2py(v) for k, v in sdr1.items()}
    # print('MRE:', mean1)
    # print('STD:', std1)
    # print('SDR:')
    # for k in sorted(sdr1.keys()):
    #     print('     {}: {}'.format(k, sdr1[k]))
    return summary

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    print(optimal_threshold)
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def auc_curve(index_name,y,prob):
    sns.set(font_scale=1.2)
    plt.rc('font', family='Times New Roman')
    fpr, tpr, thresholds = metrics.roc_curve(y,prob)
    roc_auc = metrics.auc(fpr,tpr) ###计算auc的值
    lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    print(optimal_point)
    return optimal_th

def evaluate_Trans(model, dataloader):
    # 加载模型
    model.eval()
    # if (config['checkout'] != ''):
    #    net.load_state_dict(torch.load(config['checkout']))
    dic = {}
    summary = {}
    aucs = []
    accs = []
    recalls = []
    f1s=[]
    sensitivities = []
    specificities = []
    train_loss = []
    distance_list = []
    gt_point_group = []
    class_label_group = []
    class_criterion = torch.nn.BCELoss(reduction='sum')
    for i, (images, info) in enumerate(dataloader):
        images = Variable(images).float().cuda()
        #gt = Variable(info["keypoints"]).float().cuda()
        #gt_point = gt.cpu().data.numpy().reshape((24, 2))
        #label = torch.as_tensor(info["label"], dtype=int).tolist()[0]
        label = torch.as_tensor(info["label"], dtype=int)
        gt = Variable(info["keypoints"]).float().cuda()
        size = 224
        ori_size = info["obj_origin_hw"][0].cpu().data.numpy()
        gt_point = gt.cpu().data.numpy().reshape((-1, 2))*size
        class_label = model.forward(images)
        logits = class_label['pred_logits'].squeeze(dim=-1)
        coords = class_label['pred_boxes']

        gt_kp = info["key"]
        #logits = torch.max(logits .reshape(-1, 6, 4), dim=2).values
        #label = torch.max(info["label"].reshape(-1, 6, 4), dim=2).values

        class_loss = class_criterion(logits, info["label"].float().cuda())
        # coord_loss = coord_criterion(class_output['pred_boxes'], info["keypoints"].float().cuda())

        demo_pred_poins = coords[0] .cpu().numpy() * size
        # plt.figure(2)
        # demo_img = images[0].cpu().data.numpy().reshape((size, size))
        # for j in range(24):
        #
        #
        #     plt.imshow(demo_img, cmap=plt.get_cmap('gray'))
        #
        #
        #     ##scatter
        #     plt.scatter(demo_pred_poins[j][0], demo_pred_poins[j][1], color='g',s=10)
        #     plt.scatter(gt_point[j][0], gt_point[j][1], color='r', s=10)
        # plt.show()


        cur_distance_list = cal_all_distance(gt_point, demo_pred_poins, info["loss_mask"].squeeze())
        x = 0
        length=len(cur_distance_list)
        while x< length:
            if cur_distance_list[x]==0 :
                del cur_distance_list[x]
                x-=1
                length-= 1
            x+=1

        distance_list += cur_distance_list
        dic[i] = distance_list


        train_loss.append(class_loss.item())

        if i==0:

            # pred_label_all = class_label_group
            # label_gt_all = gt_point_group

            pred_label_all = logits
            label_gt_all = label
        else:
            pred_label_all = torch.cat([pred_label_all, logits])
            label_gt_all = torch.cat([label_gt_all, label])

            # pred_label_all = torch.cat([pred_label_all, class_label_group])
            # label_gt_all = torch.cat([label_gt_all, gt_point_group])




        #print('pred',demo_pred_poins)
        #print('gt',gt)
        #calculate acc

        #print(acc)
        #计算距离


        # 评价指标

    ##divided by group
    #print(pred_label_all[0])
    pred_label_all = torch.max(pred_label_all.reshape(-1,6,4),dim=2).values
    #print(pred_label_all_group[0])
    label_gt_all = torch.max(label_gt_all.reshape(-1,6,4),dim=2).values
    loss_mean = np.mean(np.array(train_loss))

        #
    for lab in range(pred_label_all.size(1)):
            #
        pred_bool = []
            # pred_label= pred_label_all[:, lab].tolist()
        pred_label = pred_label_all[:, lab].tolist()

            # label_gt = label_gt_all[:, lab].tolist()
        label_gt = label_gt_all[:, lab].tolist()
            # print(label_gt)

            # for pred_lab in pred_label:
            #     if pred_lab >0.2:
            #         pred_bool.append(1)
            #     else:
            #         pred_bool.append(0)
            # print(sum(label_gt))
        if sum(label_gt) != 0:
            auc = metrics.roc_auc_score(label_gt, pred_label)
            thresholds = auc_curve(index_name=lab, y=label_gt, prob=pred_label)
            for pred_lab in pred_label:
                if pred_lab > thresholds:
                    pred_bool.append(1)
                else:
                    pred_bool.append(0)
            acc = metrics.balanced_accuracy_score(label_gt, pred_bool)
            tn, fp, fn, tp = metrics.confusion_matrix(label_gt, pred_bool).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            recall = metrics.recall_score(label_gt, pred_bool)
            f1 = metrics.f1_score(label_gt, pred_bool)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            recalls.append(recall)
            f1s.append(f1)
            aucs.append(auc)
            accs.append(acc)
    mean_f1 = np.around(np.mean(np.array(f1s)), decimals=4)
    mean_recall = np.around(np.mean(np.array(recalls)), decimals=4)
    mean_sensitivity = np.around(np.mean(np.array(sensitivities)), decimals=4)
    mean_specificity = np.around(np.mean(np.array(specificities)), decimals=4)
    mean_auc = np.around(np.mean(np.array(aucs)), decimals=4)
    mean_acc = np.around(np.mean(np.array(accs)), decimals=4)

    print("aucs:", aucs)
    print("accs:", accs)
    print("recalls", recalls)
    print("specificity", specificities)
    print("f1", f1s)

    print("mean_aucs:", mean_auc)
    print("mean_accs", mean_acc)
    print("mean_recall", mean_recall)
    print("mean_specificity", mean_specificity)
    print("mean_f1", mean_f1)
        #plt.show()
    li_total = []
    for d, cur_distance_list in dic.items():
        summary[d] = analysis(cur_distance_list)
        li_total += cur_distance_list
    summary['total'] = analysis_all(li_total)
    return dic, summary, mean_auc,loss_mean



def evaluate_one(model, dataloader):
    # 加载模型
    model.eval()
    # if (config['checkout'] != ''):
    #    net.load_state_dict(torch.load(config['checkout']))
    dic = {}
    summary = {}
    aucs = []
    accs = []
    recalls = []
    f1s=[]
    sensitivities = []
    specificities = []
    distance_list = []
    gt_point_group = []
    class_label_group = []
    for i, (images, info) in enumerate(dataloader):
        images = Variable(images).float().cuda()
        #gt = Variable(info["keypoints"]).float().cuda()
        #gt_point = gt.cpu().data.numpy().reshape((24, 2))
        #label = torch.as_tensor(info["label"], dtype=int).tolist()[0]
        label = torch.as_tensor(info["label"], dtype=int)

        class_label = model.forward(images)





        if i==0:

            # pred_label_all = class_label_group
            # label_gt_all = gt_point_group

            pred_label_all = class_label
            label_gt_all = label
        else:
            pred_label_all = torch.cat([pred_label_all, class_label])
            label_gt_all = torch.cat([label_gt_all, label])

            # pred_label_all = torch.cat([pred_label_all, class_label_group])
            # label_gt_all = torch.cat([label_gt_all, gt_point_group])




        #print('pred',demo_pred_poins)
        #print('gt',gt)
        #calculate acc

        #print(acc)
        #计算距离


        # 评价指标

    ##divided by group
    #print(pred_label_all[0])
    #pred_label_all_group = torch.max(pred_label_all.reshape(243,6,4),dim=2).values
    #print(pred_label_all_group[0])
    label_gt_all = torch.max(label_gt_all.reshape(-1,6,4),dim=2).values


        #
    for lab in range(pred_label_all.size(1)):
            #
        pred_bool = []
            # pred_label= pred_label_all[:, lab].tolist()
        pred_label = pred_label_all[:, lab].tolist()

            # label_gt = label_gt_all[:, lab].tolist()
        label_gt = label_gt_all[:, lab].tolist()
            # print(label_gt)

            # for pred_lab in pred_label:
            #     if pred_lab >0.2:
            #         pred_bool.append(1)
            #     else:
            #         pred_bool.append(0)
            # print(sum(label_gt))
        if sum(label_gt) != 0:
            auc = metrics.roc_auc_score(label_gt, pred_label)
            thresholds = auc_curve(index_name=lab, y=label_gt, prob=pred_label)
            for pred_lab in pred_label:
                if pred_lab > thresholds:
                    pred_bool.append(1)
                else:
                    pred_bool.append(0)
            acc = metrics.balanced_accuracy_score(label_gt, pred_bool)
            tn, fp, fn, tp = metrics.confusion_matrix(label_gt, pred_bool).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            recall = metrics.recall_score(label_gt, pred_bool)
            f1 = metrics.f1_score(label_gt, pred_bool)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            recalls.append(recall)
            f1s.append(f1)
            aucs.append(auc)
            accs.append(acc)
    mean_f1 = np.around(np.mean(np.array(f1s)), decimals=4)
    mean_recall = np.around(np.mean(np.array(recalls)), decimals=4)
    mean_sensitivity = np.around(np.mean(np.array(sensitivities)), decimals=4)
    mean_specificity = np.around(np.mean(np.array(specificities)), decimals=4)
    mean_auc = np.around(np.mean(np.array(aucs)), decimals=4)
    mean_acc = np.around(np.mean(np.array(accs)), decimals=4)

    print("aucs:", aucs)
    print("accs:", accs)
    print("recalls", recalls)
    print("specificity", specificities)
    print("f1", f1s)

    print("mean_aucs:", mean_auc)
    print("mean_accs", mean_acc)
    print("mean_recall", mean_recall)
    print("mean_specificity", mean_specificity)
    print("mean_f1", mean_f1)
        #plt.show()

    return dic, summary, mean_auc

if __name__ == '__main__':
    from dataloader_class import KFDataset
    from Spine_transformer import SpineTransformer, build
    import transform_new
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--keypoint_model", type=str, default='Spine_T',
                        help="the model name"
                             "ResNet50"
                             "DenseNet121"
                             "SENet101 "
                             "2019MICCAI")

    # parser.add_argument("--data_dir", type=str, default='../data',
    #                     help="the data dir")

    parser.add_argument("--sigma", type=float, default=10.0,
                        help="the sigma of generated heatmaps.")
    parser.add_argument("--seed", type=int, default=0,
                        help="the sigma of generated heatmaps.")

    parser.add_argument('--keypoint_batch_size', type=int, default=16,
                        help="The batch size, default: 4")
    parser.add_argument("--keypoint_model_dir", type=str, default='./Checkpoints_final/',
                        help="saving keypoint model_dir")

    parser.add_argument('--keypoint_learning_rate', type=float, default=5e-5
                        ,
                        help="The initial learning rate, default: 5e-3")

    parser.add_argument('--weights', type=str,
                        default='/public/huangjunzhang/Facetjoints/ViTweight/vit_base_patch16_224_in21k.pth',
                        help="load pretrained weight")

    ## transformer
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.5, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=24, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    args = parser.parse_args()
    data_transforms = {
        "train": transform_new.Compose([transform_new.ToTensor(),
                                     transform_new.RandomHorizontalFlip(0.5)
                                        ]),
        "val" : transform_new.Compose([transform_new.ToTensor()])
    }
    valDataset = KFDataset(config , mode='val', transforms=data_transforms["val"])
    print(len(valDataset))
    valDataLoader = DataLoader(valDataset, 1, False, num_workers=8)

    #model = UNet_Pretrained(3,24)
    #model = U2Net(in_channels=1, out_channels=24)
    #model =models.resnet50 (pretrained=True,num_classes=6)
    model , criterion, postprocessors = build(args)
    model.float().cuda()
    model.load_state_dict(torch.load("/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/Spine/SpineT_netvit_0_best_model.ckpt"))
    with torch.no_grad():
        evaluate_Trans(model=model, dataloader=valDataLoader)
