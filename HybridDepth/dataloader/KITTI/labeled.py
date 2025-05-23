
"""A loader for the labeled KITTI EIGEN dataset."""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms_v2
from PIL import Image, ImageOps
import imageio
import os
from . import preprocess
from .preprocess import CropParams, get_white_border, get_black_border
import pathlib

class KITTIEIGENLabeledDataset(Dataset):
    """Python interface for the labeled subset of the KITTI EIGEN dataset.

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, path, img_size=(480, 640), stage='test', remove_white_border=True):
        """Opens the labeled dataset file at the given path."""
        self.root_dir = path
        self.stage = stage
        """Opens the labeled dataset file at the given path."""
        with open(f'dataloader/KITTI/input_splits/eigen_split_{stage}_files_with_gt_v2.txt','r') as f:
                self.samples_pth= f.readlines()

        self.img_size = img_size
        self.remove_white_border = remove_white_border

    # def close(self):
    #     """Closes the HDF5 file from which the dataset is read."""
    #     self.file.close()

    def __len__(self):
        return len(self.samples_pth)

    def __getitem__(self, idx):

        if self.stage == 'test':
            file_path = self.samples_pth[idx].split()[0]
            p = pathlib.Path(file_path)
            base_dir = p.parts[0]
            color_img = Image.open(os.path.join(self.root_dir, self.stage,self.samples_pth[idx].split()[0]))
            depth_gt = Image.open(os.path.join(self.root_dir, self.stage, base_dir, self.samples_pth[idx].split()[1]))
        else:
            color_img_path = self.root_dir
            file_path = self.samples_pth[idx].split()[0]
            p = pathlib.Path(file_path)
            base_dir = p.parts[0]
            color_img = Image.open(f'{color_img_path}/{self.stage}/{self.samples_pth[idx].strip().split()[0]}')
            depth_gt = Image.open(f'{color_img_path}/{self.stage}/{base_dir}/{self.samples_pth[idx].strip().split()[1]}')
            

        if self.stage == 'train':
            crop_params = get_white_border(np.array(color_img, dtype=np.uint8))
        else:
            crop_params = get_black_border(np.array(color_img, dtype=np.uint8))
        
        # color_img = color_img.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
        # depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

        
        h, w = self.img_size
        # Use reflect padding to fill the blank
        color_img = np.array(color_img)
        # color_img = np.pad(color_img, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
        color_img = Image.fromarray(color_img)

        depth_gt = np.array(depth_gt)
        # depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
        depth_gt = Image.fromarray(depth_gt)
        
        color_img = np.asarray(color_img, dtype=np.float32)
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)
        # print(depth_gt.max())
        # print(depth_gt.min())
        depth_gt = depth_gt / 256.0

        # remove the borders from depth map since it can cause problem in focal stack creation
        # new_width = 1242

        # Calculate the new height to maintain the aspect ratio
        # aspect_ratio = h / w
        # new_height = int(new_width * aspect_ratio)

        # # Calculate the cropping position
        # start_row = int((h - new_height) / 2)
        # start_col = int((w - new_width) / 2)

        # # Crop the image
        # color_img = color_img[start_row:start_row + new_height, start_col:start_col + new_width]
        # depth_gt = depth_gt[start_row:start_row + new_height, start_col:start_col + new_width]
                
        t_resize = transforms_v2.Resize(self.img_size)

        color_img = color_img.transpose(2, 0, 1)
        color_img = torch.from_numpy(color_img).float()
        color_img = t_resize(color_img)

        depth_gt = depth_gt.transpose(2, 0, 1)
        depth_gt = torch.from_numpy(depth_gt).float()
        # print(depth_gt.max())
        # print(depth_gt.min())
        
        depth_gt = t_resize(depth_gt)
        return color_img, depth_gt