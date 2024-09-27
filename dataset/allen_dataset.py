import sys
sys.path.append('/content/drive/MyDrive/StableDiffusion-PyTorch-main')
import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import torch

class AllenDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for allen images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    # 默认condition_config为None，即无条件生成图像。
    # 如果不为None，condition_config设置成一个字典。本例子中，condition_config = {'condition_types': 'neuron'}。
    def __init__(self, split, im_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        
        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        
        # Conditioning for the dataset
        # 根据condition_config的值决定是否从配置中获取条件类型列表
        # condition_config是None就返回一个空列表，反之赋值为condition_config字典中'condition_types'键对应的值
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)
        
        # Whether to load images and call vae or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
        
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        # Initialize lists to store image file paths and labels.
        ims = []
        labels = []
        
        # 取出当前im_path下的所有文件名，将图片文件名存在ims_name列表里，将embedding_file单独存放（此方法要求图片和embedding_file.csv放在同一文件夹里）
        ims_name = os.listdir(im_path)
        if 'neuron' in self.condition_types:
            embedding_file = [x for x in ims_name if x.endswith('csv')]
            embedding_file = embedding_file[0]
            ims_name.remove(embedding_file)
            
            # 将embedding结果存在DataFrame里
            embedding = pd.read_csv(im_path+'/'+embedding_file)
            # 去掉118这张图片，是空白图片
            embedding = embedding[embedding['frame'] != 118].reset_index(drop = True)
            embedding.drop(columns = ['frame'], inplace = True)
        
        # 将所有文件名的后缀去掉
        ims_name_no_ext = [os.path.splitext(file)[0] for file in ims_name]
        for d_name in tqdm(ims_name_no_ext):
            # glob.glob()方法会生成一个列表，因为此处我们的列表里只有1个元素，所以直接取这个列表第一个元素，就是图片路径了
            fname = glob.glob(im_path + '/' + d_name + '.png')[0]
            # Add each image file path to the list.
            ims.append(fname)
            if 'neuron' in self.condition_types:
                labels.append(np.array(embedding.loc[int(d_name)].to_list()))

        # # Class Conditional时数据加载代码块================================================
        # ims_name = os.listdir(im_path)
        # for d_name in tqdm(ims_name):
        #     fnames = glob.glob(os.path.join(im_path, d_name, '*.png'))
        #     # Add each image file path to the list.
        #     for fname in fnames:
        #         ims.append(fname)
        #         #if 'class' in self.condition_types:
        #         labels.append(int(d_name))

        # Print the total number of images loaded for the current split.
        print('Found {} images for split {}'.format(len(ims), self.split))
        # print('标签:')
        # print(labels)
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        # 生成一个字典存储condition信息，比如可能cond_inputs = {'class': 0}
        cond_inputs = {}
        if 'neuron' in self.condition_types:
            cond_inputs['neuron'] = self.labels[index]
            cond_inputs['neuron'] = torch.tensor(self.labels[index]).unsqueeze(1)

        #######################################

        # 检查是否使用潜伏变量
        if self.use_latents:
            # 根据图像索引获取对应的潜伏变量
            latent = self.latent_maps[self.images[index]]
            # 如果没有条件类型，则只返回潜伏变量
            if len(self.condition_types) == 0:
                return latent
            # 如果有条件类型，返回潜伏变量和条件输入的元组
            else:
                return latent, cond_inputs
        else:
            # 打开并转换图像为张量
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.ToTensor()(im)
            
            # Convert input to -1 to 1 range.
            # 这里有个问题，它默认打开后图片像素点取值范围为0~1.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                # print('没标签')
                return im_tensor
            else:
                # print('有标签')
                return im_tensor, cond_inputs
            
