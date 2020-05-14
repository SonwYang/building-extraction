#!/usr/bin/env python
# coding: utf-8
import collections
import torch
import torch.nn as nn
import torch.backends.cudnn
import argparse
from utils.datasets import build_loader
from weight_loss import SoftDiceLoss, WeightedBceLoss
from utils.metrics import Metrics
import segmentation_models_pytorch as smp
from models.ternausnets import AlbuNet
from unet256.YpUnet_AG import UNet
from radam import RAdam
import os
import tqdm
import json
import datetime
import numpy as np
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(69)


def get_model(num_classes, model_name):
    if model_name == "UNet":
        print("using UNet")
        model = smp.Unet(encoder_name='resnet50', classes=num_classes, activation='sigmoid')
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
    elif model_name == "AlbuNet":
        print("using AlbuNet")
        model = AlbuNet(pretrained=True, num_classes=num_classes)
        return model
    elif model_name == "YpUnet":
        print("using YpUnet_AG")
        model = UNet(num_classes=num_classes)
        return model
    else:
        print("error in model")
        return None
    # model.train()
    # return model.to(device)

def train(loader, num_classes, device, net, optimizer, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.train()
    for images, masks in tqdm.tqdm(loader):
        images = torch.squeeze(images.to(device, dtype=torch.float))
        masks = torch.squeeze(masks.to(device))
        # print("images'size:{},masks'size:{}".format(images.size(),masks.size()))


        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, masks)
        loss.backward()
        batch_loss = loss.item()
        optimizer.step()

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
        images = torch.squeeze(images.to(device, dtype=torch.float))
        masks = torch.squeeze(masks.to(device).long())

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))

        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

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
    outPath = f"{args.results}_4c_unet_AG"
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(outPath, "{}_run_{}.json".format(args.model_name,ts))
    ##############choose model##########################
    net = get_model(args.num_classes, args.model_name).to(device)


    if args.pre_train:
        net = torch.load(args.ckp)["model_state"] #load the pretrained model
        print("load pre-trained model sucessfully")
    if torch.cuda.device_count() > 1:
        print("using multi gpu")
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    else:
        print('using one gpu')

    ##########hyper parameters setting#################
    # optimizer = Adam(net.parameters(), lr=args.lr)
    optimizer = RAdam(params=net.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=4, verbose=True)
    # milestones = [x*40 for x in range(10)]
    # print(milestones)
    # scheduler = CyclicCosAnnealingLR(optimizer,milestones=milestones,eta_min=1e-7)

    # criterion = FocalLoss2d().to(device)
    # criterion = BCEDiceLossWeighted().to(device)
    criterion = WeightedBceLoss().to(device)
    criterion2 = SoftDiceLoss().to(device)

    ##########prepare dataset################################
    train_loader, val_loader = build_loader(batch_size = args.batch_size, num_workers = 4)
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
        val_hist = validate(val_loader, args.num_classes, device, net, scheduler, criterion2)
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
    parse.add_argument("--model_name", type=str, default="YpUnet", help='myUnet, YpUnet, AlbuNet, FPN, UNet , PSPNet')
    parse.add_argument("--num_workers", type=int, default=4)
    parse.add_argument("--num_channels", type=int, default=4)

    parse.add_argument("--target_size", type=int, default=256)
    parse.add_argument("--batch_size", type=int, default=32)
    parse.add_argument("--num_epochs", type=int, default=80)
    parse.add_argument("--results", type=str, default="./results", help="the directory of model saved")
    parse.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parse.add_argument("--ckp", type=str, default="./results_AlbuNet_rgb33/AlbuNet_val_0.8492694069986265_epoch7.pth", help='the path of model weight file')
    parse.add_argument("--pre_train", type=bool, default=False, help="load the pre-trained model or not")
    parse.add_argument("--debug", type=bool, default=False, help="debug")
    args = parse.parse_args()
    main()





