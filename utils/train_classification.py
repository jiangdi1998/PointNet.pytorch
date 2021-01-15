from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size') #终端键入batchsize
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size') #默认的数据集每个点云是2500个点
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4) #进程
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for') #epoch，训练多少个权重文件
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path') #预训练模型路径
parser.add_argument('--dataset', type=str, required=True, help="dataset path") #数据集路径
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40") #数据集类型shapenet或者modelnet40
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m' #test设置成蓝色字体

opt.manualSeed = random.randint(1, 10000)  # fix seed #生成随机数
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet': #创建针对shapenet数据集的类对象
    dataset = ShapeNetDataset( #训练集
        root=opt.dataset,
        classification=True,  #打开分类的选项
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset( #测试集
        root=opt.dataset,
        classification=True,
        split='test', #标记为测试
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':#创建针对modelnet数据集的类对象
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type') #如果在终端没有键入正确的数据集格式，则警告


dataloader = torch.utils.data.DataLoader( #加载数据
    dataset,
    batch_size=opt.batchSize,
    shuffle=True, #随机数
    drop_last=True, #训练的数据数不能被batch_size整除，是不会报错的
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        drop_last=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset)) # 12137 2874
num_classes = len(dataset.classes)
print('classes', num_classes) #classes 16

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform) #调用model.py的PointNetCls定义分类函数

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model)) #如果有预训练模型，将预训练模型加载


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999)) #优化函数，可以替换成SGD之类的
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize #batch数目

for epoch in range(opt.nepoch): #在一个epoch下
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data #读取待训练对象点云与标签
        target = target[:, 0]
        points = points.transpose(2, 1) #放射变换
        points, target = points.cuda(), target.cuda() #使用cuda加速
        optimizer.zero_grad() #清除梯度
        classifier = classifier.train() #训练
        pred, trans, trans_feat = classifier(points) #计算预测值
        loss = F.nll_loss(pred, target) #交叉熵函数
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward() #反向传播
        optimizer.step() #优化
        pred_choice = pred.data.max(1)[1] #
        print(pred_choice) #tensor([ 0, 15,  4,  0], device='cuda:0')
        correct = pred_choice.eq(target.data).cpu().sum()
        print(correct) #tensor(1)
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize))) [0: 9/3034] train loss: 2.467071 accuracy: 0.250000

        if i % 10 == 0: #每十次测试以下，过程同上
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize))) #[0: 0/3034] test loss: 2.739059 accuracy: 0.000000

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch)) #保存权重文件在cls/cls_model_1.pth

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)): #tqdm进度条
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset))) #测试最终的正确率
