#!/usr/bin/env python
# coding: utf-8
import collections
import torch
import torch.nn
import torch.backends.cudnn
import argparse
from utils.datasets import build_loader
from utils.metrics import Metrics
from weight_loss import SoftDiceLoss, WeightedBceLoss
from models.YpUnet_AG_hnn234 import UNet
from radam import RAdam
import os
import tqdm
import json
import datetime
import numpy as np
import random

seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
global_step = 0


def get_model(num_classes, model_name):
    if model_name == "UNet":
        print("using UNet")
        model = smp.Unet(encoder_name='resnet101', classes=num_classes, activation='softmax')
        if args.num_channels >3:
            weight = model.encoder.conv1.weight.clone()
            model.encoder.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                print("using 4c")
                model.encoder.conv1.weight[:, :3] = weight
                model.encoder.conv1.weight[:, 3] = model.encoder.conv1.weight[:, 0]
        return model
    elif model_name == "PSPNet":
        print("using PSPNet")
        model = smp.PSPNet(encoder_name="resnet50", classes=num_classes, activation='softmax')
        if args.num_channels > 3:
            weight = model.encoder.conv1.weight.clone()
            model.encoder.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                print("using 4c")
                model.encoder.conv1.weight[:, :3] = weight
                model.encoder.conv1.weight[:, 3] = model.encoder.conv1.weight[:, 0]
        return model
    elif model_name == "FPN":
        print("using FPN")
        model = smp.FPN(encoder_name='resnet50', classes=num_classes)
        if args.num_channels > 3:
            weight = model.encoder.conv1.weight.clone()
            model.encoder.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                print("using 4c")
                model.encoder.conv1.weight[:, :3] = weight
                model.encoder.conv1.weight[:, 3] = model.encoder.conv1.weight[:, 0]
        return model
    elif model_name == "YpUNet":
        print("using YpUNet_hnn_AG")
        model = UNet(num_classes=num_classes)
        return model
    else:
        print("error in model")
        return None

def train(loader, num_classes, device, net, optimizer, criterion):
    global global_step

    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.train()
    for images, masks in tqdm.tqdm(loader):
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device)
        # print("images'size:{},masks'size:{}".format(images.size(),masks.size()))

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs, dsv4, dsv3, dsv2 = net(images)

        l2 = criterion(dsv2, masks)
        l3 = criterion(dsv3, masks)
        l4 = criterion(dsv4, masks)
        loss_fuse = criterion(outputs, masks)
        loss = (l2 + l3 + l4 + loss_fuse)/4

        loss.backward()
        batch_loss = loss.item()
        optimizer.step()

        global_step = global_step + 1

        running_loss += batch_loss

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    assert num_samples > 0, "dataset contains training images and labels"

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }

def validate(loader, num_classes, device, net, scheduler, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.eval()

    for images, masks in tqdm.tqdm(loader):
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device)

        num_samples += int(images.size(0))

        outputs = net(images)

        loss = criterion(outputs, masks)

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output)

    assert num_samples > 0, "dataset contains validation images and labels"

    scheduler.step(metrics.get_miou())  # update learning rate

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }


def main():
    outPath = f"{args.results}_{args.model_name}_YpUNet_hnn_AG"
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(outPath, "{}_run_{}.json".format(args.model_name,ts))
    ##############choose model##########################
    net = get_model(args.num_classes, args.model_name).to(device)


    if args.pre_train:
        net = torch.load(args.ckp)["model_state"] #load the pretrained model

    if torch.cuda.device_count() > 1:
        print("using multi gpu")
        net = torch.nn.DataParallel(net,device_ids = [0,1,2,3])
    else:
        print('using one gpu')

    ##########hyper parameters setting#################
    # optimizer = Adam(net.parameters(), lr=args.lr)
    optimizer = RAdam(params=net.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=5, verbose=True)
    # milestones = [x*40 for x in range(10)]
    # print(milestones)
    # scheduler = CyclicCosAnnealingLR(optimizer,milestones=milestones,eta_min=1e-7)

    criterion = WeightedBceLoss().to(device)
    # criterion = BCEDiceLossWeighted().to(device)
    # criterion = WeightedCrossEntropy2d().to(device)
    # criterion2 = DiceLoss().to(device)

    ##########prepare dataset################################
    train_loader, val_loader = build_loader(batch_size = args.batch_size, num_workers = 0)

    history = collections.defaultdict(list)

    best_miou = -100


    for epoch in range(args.num_epochs):
        print("Epoch: {}/{}".format(epoch + 1, args.num_epochs))
        # optimizer.step()
        # scheduler.step(epoch)
        ####################train####################################
        train_hist = train(train_loader, args.num_classes, device, net, optimizer, criterion)
        print( 'loss',train_hist["loss"],
                'miou',train_hist["miou"],
                'fg_iou',train_hist["fg_iou"],
                'mcc',train_hist["mcc"])

        for k, v in train_hist.items():
            history["train " + k].append(v)

    ######################valid##################################
        val_hist = validate(val_loader, args.num_classes, device, net, scheduler, criterion)
        print('loss',val_hist["loss"],
                'miou',val_hist["miou"],
                'fg_iou',val_hist["fg_iou"],
                'mcc',val_hist["mcc"])

        if val_hist["miou"] > best_miou:
            state = {
                "epoch": epoch + 1,
                "model_state": net,
                "best_miou": val_hist["miou"]
            }
            checkpoint = f'{args.model_name}_val_{val_hist["miou"]}_epoch{epoch + 1}.pth'
            torch.save(state, os.path.join(outPath, checkpoint))  # save model
            print("The model has saved successfully!")
            best_miou = val_hist["miou"]

        for k, v in val_hist.items():
            history["val " + k].append(v)

        f = open(file_path, "w+")
        f.write(json.dumps(history))
        f.close()


if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Using {}".format(device))
    parse = argparse.ArgumentParser()
    parse.add_argument("--path", type=str, default="./data", help='the root of images')
    parse.add_argument("--num_classes", type=int, default=1, help='the number of class')
    parse.add_argument("--model_name", type=str, default="YpUNet", help='Unet, YpUNet, AsppAlbuNet, AlbuNet,FPN, UNet , PSPNet')
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--num_channels", type=int, default=3)

    parse.add_argument("--target_size", type=int, default=256)
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--num_epochs", type=int, default=80)
    parse.add_argument("--results", type=str, default="./results", help="the directory of model saved")
    parse.add_argument("--lr",default=0.005,type=float,help="learning rate")
    parse.add_argument("--ckp", type=str, default="./results/segnetweights_best.pth", help='the path of model weight file')
    parse.add_argument("--pre_train", type=bool, default=False, help="load the pre-trained model or not")
    parse.add_argument("--debug", type=bool, default=False, help="debug")
    args = parse.parse_args()
    main()





