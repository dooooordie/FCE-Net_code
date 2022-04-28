from PIL import Image
import tifffile as tiff
import json, os, time, cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import skimage.util


def create_data_file(root):
    assert os.path.exists(root), "File root: {} have not exist!".format(root)
    # 获取文件名称并按文件名称排序
    file_name = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
    file_name.sort(key=lambda x:int(x))

    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF"]
    Amp = []
    Phase = []
    for name in file_name:
        # 获取图片名称并按文件名称排序
        file_path = os.path.join(root, name)
        indice = os.listdir(file_path)
        indice.sort(key=lambda x:int(x[:-6]))
        images = [os.path.join(root, name, num) for num in indice
                  if os.path.splitext(num)[-2] in supported]
        for i in range(len(images)):
            if i % 2 == 0:
                Amp.append(images[i])
            else:
                Phase.append(images[i])
    print("Here are {} Amp images in datasets".format(len(Amp)))
    print("Here are {} Phase images in datasets".format(len(Phase)))
    return Amp, Phase


def crop_images(file_root, file_name, save_path, image_size=2048, crop_size=512, thred=False):
    count = 1
    start=time.time()
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF"]
    file_path = os.listdir(os.path.join(file_root, file_name))
    file_path.sort(key=lambda x: int(x[:-4]))
    # a = os.path.splitext(file_path[1])[-1]
    images = [os.path.join(file_root, file_name, num) for num in file_path
              if os.path.splitext(num)[-1] in supported]
    num_crop = int(image_size / crop_size)

    print("Here are {} images have been cropped".format(len(images)))
    for num in range(len(images)//5):
        img = tiff.imread(images[num])
        for h in range(num_crop):
            for w in range(num_crop):
                cropped = img[h * crop_size : (1 + h) * crop_size, w * crop_size : (1 + w) * crop_size]
                i_m = cropped.max()
                if thred:
                    if i_m > 80:
                        tiff.imwrite(os.path.join(save_path, str(count) + ".tif"), cropped)
                        count += 1
                else:
                    tiff.imwrite(os.path.join(save_path, str(count) + ".tif"), cropped)
                    count += 1
        if num % (len(images) // 100) == 0:
            print("Process {:.1f}% have been completed".format(num / len(images) * 100))
    end = time.time()
    print('Running time: {:.2f}s Seconds'.format(end-start))
    print("Here are total {} cropped images".format(len(images) * (num_crop ** 2)))

def data2image(data, dtype = None):
    data[data < 0] = 0
    data[data > 1] = 1
    if dtype is None:
        dtype = 'uint8'
    if dtype is not None:
        if dtype == 'uint8':
            scale = 255
        elif dtype == 'uint16':
            scale = 65533
        else:
            scale = 1

    data = data *scale
    image = data.astype(dtype)
    return image

def write_figure(image):
    Img = image.cpu().detach().numpy().transpose(0, 2, 3, 1)
    size = list(np.shape(Img))
    num, h, w, c= size[0], size[1], size[2], size[3]
    image_list = np.split(Img, num, axis=0)
    Image1 = np.hstack((image_list[0], image_list[1]))
    Image2 = np.hstack((image_list[2], image_list[3]))
    Image = np.concatenate((Image1, Image2), axis=2).squeeze()
    Image = data2image(Image, dtype="uint8")
    return Image

def plot_loss(loss, file_save=''):
    train_loss = np.array(loss[0])
    test_loss = np.array(loss[1])
    x = np.linspace(1, len(loss[0]), len(train_loss))
    plt.title('Training and Testing loss')
    plt.xlabel('epoch', fontsize=12, color='black')
    plt.ylabel('loss', fontsize=12, color='black')
    line_train = plt.plot(x, train_loss, '-', label='train')
    line_test = plt.plot(x, test_loss, 'o--', label='test')
    plt.legend(loc=1)
    name = os.path.join(file_save, 'loss.jpg')
    plt.savefig(name)
    np_loss = np.array(loss)
    name_npy = os.path.join(file_save, 'loss.npy')
    np.save(name_npy, np_loss)
    plt.close()

def batch_PSNR(img, pred, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Pred = pred.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Pred[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

if __name__ == "__main__":
    Img = tiff.imread("G:\Python files\org.tif")
    image = skimage.util.random_noise(Img, mode='gaussian')