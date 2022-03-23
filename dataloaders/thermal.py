from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random
import cv2

class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image):
        image = np.asarray(image)
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        image = transforms.functional.to_pil_image(image)
        return image


class PairThermalDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 17
        self.void_classes = [0]
        self.valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(17)))

        self.datalist = kwargs.pop("datalist")
        self.stride = kwargs.pop('stride')
        self.iou_bound = kwargs.pop('iou_bound')

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(PairThermalDataset, self).__init__(**kwargs)

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            RandomGaussianBlur(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])
            
    def _set_files(self):
        prefix = "dataloaders/thermal_splits{}".format(self.datalist)

        if self.split == "val":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split('	') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][:][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][:][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        return image, label, image_id

    def __getitem__(self, index):

        image_path = os.path.join(self.root, self.files[index][:][1:])

        image = np.asarray(Image.open(image_path))
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][:][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        h, w, _ = image.shape

        longside = random.randint(int(self.base_size*0.8), int(self.base_size*2.0))
        h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
        image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))

        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        crop_h, crop_w = self.crop_size, self.crop_size
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,}
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_index, **pad_kwargs)

        x1 = random.randint(0, w+pad_w-crop_w)
        y1 = random.randint(0, h+pad_h-crop_h)

        max_iters = 50
        k = 0
        while k < max_iters:
            x2 = random.randint(0, w+pad_w-crop_w)
            y2 = random.randint(0, h+pad_h-crop_h)
            # crop relative coordinates should be a multiple of 8
            x2 = (x2-x1) // self.stride * self.stride + x1
            y2 = (y2-y1) // self.stride * self.stride + y1
            if x2 < 0: x2 += self.stride
            if y2 < 0: y2 += self.stride

            if (crop_w - abs(x2-x1)) > 0 and (crop_h - abs(y2-y1)) > 0:
                inter = (crop_w - abs(x2-x1)) * (crop_h - abs(y2-y1))
                union = 2*crop_w*crop_h - inter
                iou = inter / union
                if iou >= self.iou_bound[0] and iou <= self.iou_bound[1]:
                    break
            k += 1

        if k == max_iters:
            x2 = x1
            y2 = y1

        overlap1_ul = [max(0, y2-y1), max(0, x2-x1)]
        overlap1_br = [min(self.crop_size, self.crop_size+y2-y1, h//self.stride * self.stride), min(self.crop_size, self.crop_size+x2-x1, w//self.stride * self.stride)]
        overlap2_ul = [max(0, y1-y2), max(0, x1-x2)]
        overlap2_br = [min(self.crop_size, self.crop_size+y1-y2, h//self.stride * self.stride), min(self.crop_size, self.crop_size+x1-x2, w//self.stride * self.stride)]

        try:
            assert (overlap1_br[0]-overlap1_ul[0]) * (overlap1_br[1]-overlap1_ul[1]) == (overlap2_br[0]-overlap2_ul[0]) * (overlap2_br[1]-overlap2_ul[1])
            assert overlap1_br[0] >= 0 and overlap1_ul[0] >= 0 and overlap1_br[1] >= 0 and overlap1_ul[1] >= 0
            assert overlap2_br[0] >= 0 and overlap2_ul[0] >= 0 and overlap2_br[1] >= 0 and overlap2_ul[1] >= 0
        except:
            print("k: {}".format(k))
            print("h: {}, w: {}".format(h, w))
            print("image.shape: ", image.shape)
            print("x1: {}, x2: {}, y1: {}, y2: {}".format(x1, x2, y1, y2))
            print("image_path:", image_path)
            print("ul1: ", overlap1_ul)
            print("br1: ", overlap1_br)
            print("ul2: ", overlap2_ul)
            print("br2: ", overlap2_br)
            print("index: ", index)
            exit()

        image1 = image[y1:y1+self.crop_size, x1:x1+self.crop_size].copy()
        image2 = image[y2:y2+self.crop_size, x2:x2+self.crop_size].copy()
        label1 = label[y1:y1+self.crop_size, x1:x1+self.crop_size].copy()
        label2 = label[y2:y2+self.crop_size, x2:x2+self.crop_size].copy()

        try:
            assert image1[overlap1_ul[0]:overlap1_br[0], overlap1_ul[1]:overlap1_br[1]].shape == image2[overlap2_ul[0]:overlap2_br[0], overlap2_ul[1]:overlap2_br[1]].shape
        except:
            print("k: {}".format(k))            
            print("h: {}, w: {}".format(h, w))
            print("image.shape: ", image.shape)
            print("x1: {}, x2: {}, y1: {}, y2: {}".format(x1, x2, y1, y2))
            print("image_path:", image_path)
            print("ul1: ", overlap1_ul)
            print("br1: ", overlap1_br)
            print("ul2: ", overlap2_ul)
            print("br2: ", overlap2_br)
            print("index: ", index)
            exit()

        flip1 = False
        if random.random() < 0.5:
            image1 = np.fliplr(image1)
            label1 = np.fliplr(label1)
            flip1 = True

        flip2 = False
        if random.random() < 0.5:
            image2 = np.fliplr(image2)
            label2 = np.fliplr(label2)
            flip2 = True
        flip = [flip1, flip2]

        image1 = self.train_transform(image1)
        image2 = self.train_transform(image2)

        images = torch.stack([image1, image2])
        labels = torch.from_numpy(np.stack([label1, label2]))
        return images, labels, overlap1_ul, overlap1_br, overlap2_ul, overlap2_br, flip
    def decode_segmap(self,temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        colors = self.get_class_colors()
        for l in range(0, self.num_classes):
            r[temp == l] = colors[l][0]
            g[temp == l] = colors[l][1]
            b[temp == l] = colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self,mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


class PairThermal(BaseDataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')
        self.dataset = PairThermalDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)

        super(PairThermal, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None, dist_sampler=dist_sampler)


class ThermalDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 17

        self.datalist = kwargs.pop("datalist")
        self.palette = pallete.get_voc_pallete(self.num_classes)
        self.void_classes = [0]
        self.valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(17)))
        super(ThermalDataset, self).__init__(**kwargs)

    def _set_files(self):
        prefix = "dataloaders/thermal_splits{}".format(self.datalist)

        if self.split == "val":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split('	') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][:][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][:][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        return image, label, image_id

    def decode_segmap(self,temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        colors = self.get_class_colors()
        for l in range(0, self.num_classes):
            r[temp == l] = colors[l][0]
            g[temp == l] = colors[l][1]
            b[temp == l] = colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self,mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def get_class_colors(*args):
        return [[128, 64, 128],
                [244, 35, 232],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [119, 11, 32],
                [70, 70, 70],
                [190, 153, 153],
                [150, 100, 100],
                [153, 153, 153],
                [220, 220, 0],
                [250, 170, 30],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180]]

class Thermal(BaseDataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')

        self.dataset = ThermalDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)

        super(Thermal, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None, dist_sampler=dist_sampler)

