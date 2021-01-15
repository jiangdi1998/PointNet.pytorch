from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size') #batch_size
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4) #加载数据的进程数
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for') #epoch
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path') #model路径
parser.add_argument('--dataset', type=str, required=True, help="dataset path") #数据集路径
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice") #训练类别选择，例如椅子
parser.add_argument('--feature_transform', action='store_true', help="use feature transform") #使用特征放射变换

opt = parser.parse_args()
print(opt) #打印输入的信息

opt.manualSeed = random.randint(1, 10000)  # fix seed #随机数
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False, #不进行图像分类,只分割
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(  #加载数据
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,  #shuffle根据随机数，数据集打乱
    num_workers=int(opt.workers)) #workers为进程数

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test', #选取为测试模式
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset)) #打印数据集和测试集的对象个数
num_classes = dataset.num_seg_classes #从misc/num_seg_classes.txt读取分割个数
print('classes', num_classes) #例如chair的分割个数是4
try:
    os.makedirs(opt.outf) #用于递归创建目录
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m' #在训练时，将test设置成蓝色字底

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform) #读取model.py的densecls函数

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model)) #如果有预训练模型，加载预训练模型

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999)) #优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) 
classifier.cuda()

num_batch = len(dataset) / opt.batchSize #数据集分成batch的个数

for epoch in range(opt.nepoch):  #执行一个epoch
    scheduler.step()      
    for i, data in enumerate(dataloader, 0): #enumerate枚举
        points, target = data #读取数据
        points = points.transpose(2, 1) #仿射变换
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad() #梯度清除
        classifier = classifier.train() #训练
        pred, trans, trans_feat = classifier(points) #预测
        pred = pred.view(-1, num_classes)#pred:tensor([[-2.1568, -1.0128, -0.8314, -2.4574],[-2.4310, -1.1079, -0.6761, -2.6147],...,])
        target = target.view(-1, 1)[:, 0] - 1#target:tensor([1, 1, 0, ..., 1, 1, 1])
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target) #交叉熵函数
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward() #反向传播
        optimizer.step() #优化
        pred_choice = pred.data.max(1)[1]  #pred_choice:tensor([1, 1, 1,  ..., 0, 2, 0], device='cuda:0')
        correct = pred_choice.eq(target.data).cpu().sum()  #将pred_choice与target对比，计算正确率
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

        if i % 10 == 0: #每十次测试以下,过程同上
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))

    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch)) #保存在seg/seg_model_Chair_1.pth

## benchmark mIOU #mIOU = TP/(FP+FN+TP)
shape_ious = [] #测试
for i,data in tqdm(enumerate(testdataloader, 0)): #tqdm进度条 
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:  #计算IOU
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious))) #输出IOU
