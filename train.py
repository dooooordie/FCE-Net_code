import os, time, argparse, tifffile
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from Data_Loader import Create_dataloader

from models.FCENet import FCENet
from utils import write_figure, plot_loss, batch_PSNR

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    # T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.CenterCrop(512)
])


parser = argparse.ArgumentParser(description="The FCE-Net train.py")
parser.add_argument("--Train info", dest= "Train Info", type=str, default='''\n''')
parser.add_argument("--batch_size", dest= "batchsize", type=int, default=16)
parser.add_argument("--learning_rate", dest= "lr", type=float, default=1e-4)
parser.add_argument("--epoch number", dest="num_epochs", type=int, default=100)
parser.add_argument("--images direction", dest="img_dir", type=str, default=r'G:\ACEdata\noise\input')
parser.add_argument("--masks direction", dest="mask_dir", type=str, default=r'G:\ACEdata\noise\masks')
parser.add_argument("--result direction", dest="save_dir", type=str, default=r"D:\python flie\enhancement\ignite\results")
parser.add_argument("--weights file", dest="w_file", type=str, default="weights.pth")
parser.add_argument("--file name", dest="file_name", type=str, default=r"mice_noise1")
parser.add_argument("--validset ratio", dest="valid_r", type=float, default=0.1)
args = parser.parse_args()

trainloader, validloader = Create_dataloader(args.img_dir, args.mask_dir, args.valid_r,
                                             batch_size=args.batchsize, transform=transform)
total_train, total_valid = int(len(trainloader) * args.batchsize), int(len(validloader) * args.batchsize)

def Train(trainloader, model, optimizer, criterion, scheduler):
    total_loss = 0.0
    total_psnr = 0.0
    model.train()

    for batch, data in enumerate(trainloader):
        optimizer.zero_grad()
        input, mask = data[0].to(device), data[1].to(device)
        output = model(input)
        loss = criterion(output, mask)
        psnr = batch_PSNR(output, mask, 1.)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr.item()
        if batch % (total_train//args.batchsize//10) == 0:
            print("[{:}/{:}], Running loss:{:.6f}, PSNR: {:.4f}".format(batch * args.batchsize, total_train, loss, psnr))

    # scheduler.step()
    train_loss = total_loss / total_train * args.batchsize
    train_psnr = total_psnr / total_train * args.batchsize
    print("One epoch train loss: {:.6f} PSNR: {:.4f}".format(train_loss, train_psnr))
    return train_loss

def Valid(validloader, model, criterion):
    loss, psnr = 0.0, 0.0
    model.eval()
    for batch, data in enumerate(validloader):
        input, mask = data[0].to(device), data[1].to(device)
        output = model(input)
        loss += criterion(output, mask).item()
        psnr += batch_PSNR(output, mask, data_range=1.)

    img_input = write_figure(input)
    img_output = write_figure(output)
    img_mask = write_figure(mask)
    tifffile.imwrite(os.path.join(args.save_dir, args.file_name, "Input.tif"), img_input)
    tifffile.imwrite(os.path.join(args.save_dir, args.file_name, "Prediction.tif"), img_output)
    tifffile.imwrite(os.path.join(args.save_dir, args.file_name, "GroundTruth.tif"), img_mask)
    valid_loss = loss / total_valid * args.batchsize
    valid_psnr = psnr / total_valid * args.batchsize
    print("One epoch valid loss: {:.6f} PSNR: {:.4f}".format(valid_loss, valid_psnr))
    return valid_loss





def Main(num_epochs=60):
    model_loss = [[],[]]
    best_loss = 1e4
    model = FCENet(in_channel=1, out_channel=1, n_filter=64, init_weight=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    if os.path.exists(os.path.join(args.save_dir, args.file_name)) is not True:
        os.mkdir(os.path.join(args.save_dir, args.file_name))

    print('''
============Start training=================
            Epochs:                 {}
            learning rate:          {}
            Batch size:             {}
            Device:                 {}
            Trainset size:          {}
            Validset size:          {}
            '''.format(args.num_epochs, args.lr, args.batchsize, device, total_train, total_valid))

    for epoch in range(1, num_epochs+1):
        print(f'\nEpoch: {epoch}/{num_epochs}')
        print('Learning rate: {:.8f}\n|-------------------------------'.format(optimizer.param_groups[0]['lr']))
        start = time.perf_counter()
        train_loss = Train(trainloader, model, optimizer, criterion, scheduler)
        valid_loss = Valid(validloader, model, criterion)
        print("Using time in one epoch: {:.2f}min".format((time.perf_counter()-start)/60))
        model_loss[0].append(train_loss), model_loss[1].append(valid_loss)
        plot_loss(model_loss, file_save=os.path.join(args.save_dir, args.file_name))
        if best_loss > valid_loss:
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.file_name, args.w_file))
            best_loss = valid_loss
    print("Network training has already done!")
if __name__ ==  "__main__":
    Main(num_epochs=args.num_epochs)











