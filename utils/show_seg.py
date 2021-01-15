from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='/home/jiangdi/Downloads/jiangdi1998-pointnet.pytorch-master/pointnet.pytorch/utils/seg/seg_model_Chair_1.pth', help='model path')#模型路径
parser.add_argument('--idx', type=int, default=0, help='model index') #选取model，例如0/704
parser.add_argument('--dataset', type=str, default='/home/jiangdi/Downloads/jiangdi1998-pointnet.pytorch-master/pointnet.pytorch/utils/shapenetcore_partanno_segmentation_benchmark_v0', help='dataset path')#数据集的路径
parser.add_argument('--class_choice', type=str, default='Chair', help='class choice') #选择识别的类别，例如椅子

opt = parser.parse_args()
print(opt) #打印路径、类别等信息

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice], #类别选择
    split='test', #测试集
    data_augmentation=False)

idx = opt.idx

print("model %d/%d" % (idx, len(d))) #选取model，例如0/704
point, seg = d[idx]
print(point.size(), seg.size()) #输出点云点数，seg
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)#上色
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

state_dict = torch.load(opt.model) #加载模型
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval() #固定

point = point.transpose(1, 0).contiguous() #仿射变换
point = Variable(point.view(1, point.size()[0], point.size()[1])) #point.size()[0]为3，point.size()[1]为2500，view相当于resize
pred, _, _ = classifier(point) #预测
pred_choice = pred.data.max(2)[1] #pred_choice为每个点的分类，例如输出为tensor([[1, 0, 0, ..., 0, 0, 2]])
print(pred_choice)

#print(pred_choice.size()) #torch.size[1,2500]
pred_color = cmap[pred_choice.numpy()[0], :] #根据类别为每个点上不同颜色

#print(pred_color.shape)#torch.size[3,2500]
showpoints(point_np, gt, pred_color) #调用show3d_ball程序显示3d图像
