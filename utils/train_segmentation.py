from __future__ import print_function
import argparse
import os
import random
import time
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_reguliarzer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize


def get_iou(pred_np, target_np):
    shape_ious = []
    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)  # np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return np.mean(shape_ious)


best_iou = 0
for epoch in range(opt.nepoch):
    scheduler.step()
    loss_buf = []
    epoch_start_time = time.time()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        # print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        loss_buf.append(loss.detach().cpu().numpy())
        if opt.feature_transform:
            loss += feature_transform_reguliarzer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

    # finish one epoch
    print('Epoch %d: Train loss: %.2f, Time: %.2f s' % (epoch, np.mean(loss_buf), time.time() - epoch_start_time))
    # every ten epoch: save model & evaluate

    if (epoch) % 10 == 0:
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch + 1))
        # evaluate on test set
        loss_buf = []
        correct_buf = []
        iou_buf = []
        for i, data in enumerate(testdataloader, 0):
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, _, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            # calculate the mIOU
            pred_np = pred_choice.cpu().data.numpy()
            target_np = target.cpu().data.numpy()
            shape_iou = get_iou(pred_np, target_np)

            iou_buf.append(shape_iou)
            loss_buf.append(loss.data.item())
            correct_buf.append(correct.item())
        # print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct / float(32)))
        acc = np.sum(correct_buf) * 100 / test_dataset.__len__() / 2500
        print("Epoch %d: Test loss: %.4f Accuracy: %.2f%% IOU: %.2f%%" % (
            epoch, np.mean(loss_buf), acc, np.mean(iou_buf) * 100.0))
        if np.mean(iou_buf) > best_iou:
            best_iou = np.mean(iou_buf)
            torch.save(classifier.state_dict(), '%s/seg_model_%s_best.pth' % (opt.outf, opt.class_choice))
