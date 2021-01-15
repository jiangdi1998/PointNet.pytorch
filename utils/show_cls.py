from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '/home/jiangdi/Downloads/jiangdi1998-pointnet.pytorch-master/pointnet.pytorch/utils/cls/cls_model_0.pth',  help='model path') #加载模型
parser.add_argument('--num_points', type=int, default=2500, help='input batch size') #默认的每个点云点数2500


opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset( #测试集为ShapeNetDataset
    root='shapenetcore_partanno_segmentation_benchmark_v0',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader( #加载测试集数据
    test_dataset, batch_size=8, shuffle=True) #显存小，修改batch_size为8或者4

classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()


for i, data in enumerate(testdataloader, 0):
    points, target = data
    points, target = Variable(points), Variable(target[:, 0])
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred, _, _ = classifier(points)
    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    print(pred_choice) #预测  tensor([ 4, 12, 15,  4, 15, 15, 12,  6], device='cuda:0')
    print(target) #真值 tensor([ 4, 15, 15,  4, 15, 15, 15,  8], device='cuda:0')
    correct = pred_choice.eq(target.data).cpu().sum() #计算正确率 
    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct / float(8))) #显存小，修改batch_size为8或者4  ，i:0  loss: 0.426616 accuracy: 0.750000
