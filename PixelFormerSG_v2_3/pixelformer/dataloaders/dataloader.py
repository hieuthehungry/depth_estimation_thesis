import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random
import cv2

from utils import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch])
    depth = torch.stack([torch.as_tensor(b['depth']) for b in batch]) if 'depth' in batch[0] else None
    try:
        has_valid_depth = batch[0]["has_valid_depth"] 
    except:
        has_valid_depth = None
    scene_graphs = [b['scene_graph'] for b in batch]  # list of dicts

    # Example of stacking scene_graph tensors if shapes match
    # if all(sg['obj_logits'].shape == batch[0]['scene_graph']['obj_logits'].shape for sg in scene_graphs):
    if len(batch) > 1:
        """
            'pred_logits', 'pred_boxes', 'sub_logits', 'sub_boxes', 'obj_logits', 'obj_boxes', 'rel_logits'
        """
        batched_scene_graphs = {'pred_logits': None, 'pred_boxes': None, 
                                'sub_logits': None, 'sub_boxes': None, 'obj_logits': None, 
                                'obj_boxes': None, 'rel_logits': None}
        for key in batched_scene_graphs.keys():    
            batched_scene_graphs[key] = torch.stack([sg[key].squeeze() for sg in scene_graphs])
    else:
        batched_scene_graphs = scene_graphs[0]
    # Otherwise keep as list for your model's custom handling
    
    return {
        "image": images,
        "depth": depth,
        "scene_graph": batched_scene_graphs,
        "focal": torch.tensor([b['focal'] for b in batch]),
        "has_valid_depth": has_valid_depth
    }

class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler, collate_fn = collate_fn)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler,  collate_fn = collate_fn)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1,  collate_fn = collate_fn)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

def adjust_boxes(boxes, crop_box, original_size, discard_outside=True):
    """
    Adjust relative [cx, cy, w, h] boxes after a crop.

    Args:
        boxes (Tensor): shape [N, 4], values in relative format (cx, cy, w, h)
        crop_box (tuple): (top, left, height, width) of the crop (absolute pixels)
        original_size (tuple): (H, W) of original image before crop
        discard_outside (bool): whether to discard boxes that fall completely outside the crop

    Returns:
        new_boxes (Tensor): [M, 4] relative boxes inside crop (if discard_outside=True)
        mask (Tensor): [N] bool tensor indicating kept boxes (if discard_outside=True)
    """
    top, left, crop_h, crop_w = crop_box
    orig_h, orig_w = original_size

    boxes = boxes.clone()

    # Convert to absolute format: [x1, y1, x2, y2]
    cx = boxes[:, 0] * orig_w
    cy = boxes[:, 1] * orig_h
    w = boxes[:, 2] * orig_w
    h = boxes[:, 3] * orig_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Check intersection with crop
    x1_new = x1 - left
    y1_new = y1 - top
    x2_new = x2 - left
    y2_new = y2 - top

    # Calculate center and size in crop
    cx_new = (x1_new + x2_new) / 2
    cy_new = (y1_new + y2_new) / 2
    w_new = x2_new - x1_new
    h_new = y2_new - y1_new

    # Optional: filter boxes that are fully outside the crop
    keep = (x2_new > 0) & (y2_new > 0) & (x1_new < crop_w) & (y1_new < crop_h)
    if discard_outside:
        cx_new = cx_new[keep]
        cy_new = cy_new[keep]
        w_new = w_new[keep]
        h_new = h_new[keep]

    # Normalize to [0, 1] based on cropped size
    cx_new = cx_new / crop_w
    cy_new = cy_new / crop_h
    w_new = w_new / crop_w
    h_new = h_new / crop_h

    new_boxes = torch.stack([cx_new, cy_new, w_new, h_new], dim=-1)

    return new_boxes, keep if discard_outside else torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)

            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':
            if self.args.dataset == 'kitti':
                rgb_file = sample_path.split()[0]
                depth_file = os.path.join(sample_path.split()[0].split('/')[0], sample_path.split()[1])
                scene_graph_file = rgb_file.replace(".png", ".pt")
                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file.replace('image_02', 'image_03')
                    depth_file.replace('image_02', 'image_03')
            else:
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]

            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)
            scene_graph_path = os.path.join(self.args.sg_path,scene_graph_file)
    
            image = Image.open(image_path)
            scene_graph = torch.load(scene_graph_path)
            depth_gt = Image.open(depth_path)
            
            

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            
            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    valid_mask = np.zeros_like(depth_gt)
                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask==0] = 0
                    depth_gt = Image.fromarray(depth_gt)
                else:
                    depth_gt = depth_gt.crop((43, 45, 608, 472))
                    image = image.crop((43, 45, 608, 472))
    
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
                img, depth = image, depth_gt
                #<https://arxiv.org/abs/2107.07684>
                H, W = img.shape[0], img.shape[1]
                a, b, c, d = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
                l, u = int(a*W), int(b*H)
                w, h = int(max((W-a*W)*c*0.75, 1)), int(max((H-b*H)*d*0.75, 1))
                depth_copied = np.repeat(depth, 3, axis=2)
                M = np.ones(img.shape)
                M[l:l+h, u:u+w, :] = 0
                img = M*img + (1-M)*depth_copied
                image = img.astype(np.float32)
            else:
                depth_gt = depth_gt / 256.0

            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                # print("Trúc xinh trúc mọc đầu đình")
                H = self.args.input_height
                W = self.args.input_width
                b = 0
                image, depth_gt, crop_box = self.random_crop(image, depth_gt, H, W)
                sub_boxes, sub_keep = adjust_boxes(scene_graph['sub_boxes'][b], crop_box, (H, W))
                obj_boxes, obj_keep = adjust_boxes(scene_graph['obj_boxes'][b], crop_box, (H, W))

                # You must apply this mask consistently to rel_logits, sub_logits, obj_logits too:
                scene_graph['sub_boxes'] =  scene_graph['sub_boxes'][b][sub_keep & obj_keep].unsqueeze()
                scene_graph['obj_boxes'] =  scene_graph['obj_boxes'][b][sub_keep & obj_keep].unsqueeze()
                scene_graph['rel_logits'] = scene_graph['rel_logits'][b][sub_keep & obj_keep].unsqueeze()
                scene_graph['sub_logits'] = scene_graph['sub_logits'][b][sub_keep & obj_keep].unsqueeze()
                scene_graph['obj_logits'] = scene_graph['obj_logits'][b][sub_keep & obj_keep].unsqueeze()

            image, depth_gt = self.train_preprocess(image, depth_gt, scene_graph)
            sample = {'image': image, "scene_graph":  scene_graph, 'depth': depth_gt, 'focal': focal}
        
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            scene_graph_file = sample_path.split()[0].replace(".png", ".pt")
            scene_graph_path =  os.path.join(self.args.sg_path_eval,scene_graph_file)
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
            scene_graph = torch.load(scene_graph_path)
            
            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                if self.args.dataset == 'kitti':
                    depth_path = os.path.join(gt_path, sample_path.split()[0].split('/')[0], sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            if self.mode == 'online_eval':
                sample = {'image': image, "scene_graph": scene_graph, 'depth': depth_gt,  'focal': focal, 'has_valid_depth': has_valid_depth, 'path': image_path}
            else:
                sample = {'image': image,  "scene_graph": scene_graph, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)
            sample["scene_graph"] = scene_graph
            sample['path'] =  image_path
            if  self.mode == 'online_eval':
                sample["has_valid_depth"] = has_valid_depth
        # print(sample.keys())
        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    # def random_crop(self, img, depth, height, width):
    #     assert img.shape[0] >= height
    #     assert img.shape[1] >= width
    #     assert img.shape[0] == depth.shape[0]
    #     assert img.shape[1] == depth.shape[1]
    #     x = random.randint(0, img.shape[1] - width)
    #     y = random.randint(0, img.shape[0] - height)
    #     img = img[y:y + height, x:x + width, :]
    #     depth = depth[y:y + height, x:x + width, :]
    #     return img, depth
    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]

        y = random.randint(0, img.shape[0] - height)
        x = random.randint(0, img.shape[1] - width)

        img_cropped = img[y:y + height, x:x + width, :]
        depth_cropped = depth[y:y + height, x:x + width, :]

        crop_box = (y, x, height, width)  # top, left, h, w
        return img_cropped, depth_cropped, crop_box

    def train_preprocess(self, image, depth_gt, scene_graph=None):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            if scene_graph is not None:
                for b in range(len(scene_graph['sub_boxes'])):
                    scene_graph['sub_boxes'][b][:, 0] = 1.0 - scene_graph['sub_boxes'][b][:, 0]
                    scene_graph['obj_boxes'][b][:, 0] = 1.0 - scene_graph['obj_boxes'][b][:, 0]
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 'path': sample['path']}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
