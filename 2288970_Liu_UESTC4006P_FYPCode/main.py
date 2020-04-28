import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim, nn
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from mIou import *
import os
import cv2
import PIL.Image as Image

# Using cuda or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(i):
    import dataset
    imgs = dataset.make_dataset(r"/Users/apple/Desktop/FYPtest_2288970/data/val")
    imgx = []
    imgy = []
    for img in imgs:
        imgx.append(img[0])
        imgy.append(img[1])
    return imgx[i],imgy[i]


def train_model(model, criterion, optimizer, dataload, num_epochs=125):
    Loss_list=[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0

        for x, y in dataload:
            step += 1

            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients

            optimizer.zero_grad()
            # forward

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))

        Loss_list.append(epoch_loss/step)

    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)

    x=range(0,125)
    y=Loss_list
    plt.plot(x,y,'.-')
    plt.show()
    return model

#training model
def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size

    LR=0.005
    optimizer=optim.SGD(model.parameters(),
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=0.0005
                        )
    criterion = nn.BCEWithLogitsLoss()
    lr_list=[]

    liver_dataset = LiverDataset("data/train",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

#Display the output of the model
def test(args):
    model = Unet(3, 1)   #The unet input is three channels, and the output is one channel, because there is only one category of fingerprints on the background
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))  #Load the trained model
    liver_dataset = LiverDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()  #Turn on dynamic mode


    with torch.no_grad():
        i=0
        miou_total=0
        num=len(dataloaders)
        for x, _ in dataloaders:
            x=x.to(device)
            y=model(x)

            img_y=torch.squeeze(y).cpu().numpy() #Before inputting the loss function, the prediction graph must be converted into numpy format, and in order to correspond to the training graph, an additional one-dimensional representation of the batch size must be added
            mask=get_data(i)[1]  #Get the current mask path
            miou_total += get_iou(mask, img_y)  #Get the miou of the current prediction graph and add it to the total miou
            plt.subplot(121)
            plt.imshow(Image.open(get_data(i)[0]))
            plt.subplot(122)
            img_y=img_y*255
            img_y=Image.fromarray(img_y)
            plt.imshow(img_y.convert('L'))
            plt.pause(2)
            if i < num: i += 1  # Processing the next set of validation sets
        plt.show()
        print('Miou=%f'%(miou_total/106))


if __name__ == '__main__':
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    #mask only needs to be converted to tensor
    y_transforms = transforms.ToTensor()


    #Parameter analysis
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    parse.add_argument('--learning_rate', dest='lr', type=float, default=0.1, help='learning rate')
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
