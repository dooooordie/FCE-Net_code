
from torch.utils.data import DataLoader
import torch
import os, time, cv2
from models import FCENet
import tifffile as tiff
from PIL import Image
from utils import data2image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.CenterCrop(512),
])


class basic_dataset(object):
    def __init__(self, root, file_name, transform=None):
        self.root = root
        self.file_name = file_name
        self.transform = transform
        self.img = list(sorted(os.listdir(os.path.join(root, file_name))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.file_name, self.img[idx])
        imgs = tiff.imread(img_path)

        if transform is not None:
            imgs = self.transform(imgs)
        return imgs

    def __len__(self):
        return len(self.img)


def image_display(data, image_name: list, pred_path, imshow=True, imwrite=False):
    for i in range(len(data)):
        image = data[i].cpu().detach().numpy().transpose((1, 2, 0))
        image = data2image(image, dtype="uint8")

        if imwrite == True:
            pred_file = os.path.join(pred_path, image_name)
            tiff.imwrite(pred_file, image)
        if imshow == True:
            plt.imshow(image)
            # plt.show()

def image_preds(model, testloader, image_name, pred_path, device):
    t1 = time.perf_counter()
    model.eval()
    pic_num = 0
    with torch.no_grad():
        for batch, data in enumerate(testloader):
            t2 = time.perf_counter()
            inputs= data.to(device)
            outputs = model(inputs)
            image_display(outputs, image_name[pic_num], pred_path, imshow=False, imwrite=True)
            print('Running pre time: {:.4f}s Seconds'.format(time.perf_counter() - t2))
            pic_num += 1
    print('Running total time: {:.2f}s Seconds'.format(time.perf_counter() - t1))





file_root = r"G:\ACEdata\Score"
file_name = r"org"
pred_name = r"dpnet-on"
file_model = r'D:\python flie\enhancement\ignite\results\DP-large-ontune\model_weight.pth'

images = basic_dataset(file_root, file_name, transform=transform)

pred_path = os.path.join(file_root, pred_name)
# image_name.sort(key=lambda x: int(x[:-4]))

predloader = DataLoader(images, batch_size=1, shuffle=False)
image_name = predloader.dataset.img
model = FCENet.FCENet(in_channel=1, out_channel=1, n_filter=64, init_weight=False).to(device)
model.load_state_dict(torch.load(file_model, map_location=device))
print("Here are {} images have been predicted".format(len(image_name)))
image_preds(model, predloader, image_name, pred_path, device)







