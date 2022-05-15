from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import math
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random
import cv2
import copy
from augmentations import get_composed_augmentations
from memory_profiler import profile

class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image):
        image = np.asarray(image)
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        image = transforms.functional.to_pil_image(image)
        return image

class ThermalVid(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')

        self.dataset = ThermalVidDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)

        super(ThermalVid, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                         dist_sampler=dist_sampler)
class ThermalVidDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 17

        self.datalist = kwargs.pop("datalist")
        self.palette = pallete.get_voc_pallete(self.num_classes)
        self.void_classes = [0]
        self.valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(17)))
        super(ThermalVidDataset, self).__init__(**kwargs)

    def _set_files(self):
        prefix = "dataloaders/thermalvid_splits{}".format(self.datalist)

        if self.split == "val":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split == "test":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split == "train_supervised":
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_{self.split}" + ".txt")
        elif self.split == "train_unsupervised":
            file_list = os.path.join(prefix, f"1-{self.n_unlabeled_ratio}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split('	') for line in tuple(open(file_list, "r"))]
        if self.split == "train_unsupervised":
            self.files, self.labels = list(file_list), None
        else:
            self.files, self.labels = list(zip(*file_list))
        self.image_loaded = np.empty([len(file_list),self.base_size[0],self.base_size[1],3], dtype=float)
        self.label_loaded = np.empty([len(file_list), self.base_size[0], self.base_size[1]], dtype=int)
        for idx in len(self.files):
            img_path = os.path.join(self.root, self.files[idx][:][1:])
            self.image_loaded[idx] = np.asarray(Image.open(img_path), dtype=np.float32)
            label_path = os.path.join(self.root, self.labels[idx][:][1:])
            label = np.asarray(Image.open(label_path), dtype=np.uint8)
            self.label_loaded[idx] = self.encode_segmap(label)

    def _load_data(self, index):
        # image_path = os.path.join(self.root, self.files[index][:][1:])
        # image = np.asarray(Image.open(image_path), dtype=np.float32)
        # # image_id = self.files[index].split("/")[-1].split(".")[0]
        # if self.use_weak_lables:
        #     label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        # else:
        #     label_path = os.path.join(self.root, self.labels[index][:][1:])
        # if label_path is not None:
        #     label = np.asarray(Image.open(label_path), dtype=np.int32)
        #     label = self.encode_segmap(np.array(label, dtype=np.uint8))
        return self.image_loaded[index], self.label_loaded[index] #, image_id
        # else:
        #     return self.image_loaded[index], None#, image_id

    def decode_segmap(self, temp):
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

    def encode_segmap(self, mask):
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

class PairThermalVid(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')
        self.dataset = PairThermalVidDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)

        super(PairThermalVid, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                             dist_sampler=dist_sampler)
class PairThermalVidDataset(BaseDataSet):
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
        super(PairThermalVidDataset, self).__init__(**kwargs)

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            RandomGaussianBlur(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _set_files(self):
        prefix = "dataloaders/thermalvid_splits{}".format(self.datalist)

        if self.split == "val":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split == "test":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split == "train_supervised":
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_{self.split}" + ".txt")
        elif self.split == "train_unsupervised":
            file_list = os.path.join(prefix, f"1-{self.n_unlabeled_ratio}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        if self.split == "train_unsupervised":
            file_list = [line.rstrip() for line in tuple(open(file_list, "r"))]
            self.files, self.labels = file_list, None
        else:
            file_list = [line.rstrip().split('	') for line in tuple(open(file_list, "r"))]
            self.files, self.labels = list(zip(*file_list))
        # file_list = [line.rstrip() for line in tuple(open(file_list, "r"))]
        self.image_loaded = np.empty([len(file_list),self.base_size[0],self.base_size[1],3], dtype=float)
        # self.label_loaded = np.empty([len(file_list), self.base_size[0], self.base_size[1]], dtype=int)
        for idx in len(file_list):
            img_path = os.path.join(self.root, self.files[idx][1:])
            self.image_loaded[idx] = np.asarray(Image.open(img_path), dtype=np.float32)
            # label_path = os.path.join(self.root, self.labels[idx][1][1:])
            # label = np.asarray(Image.open(label_path), dtype=np.uint8)
            # self.label_loaded[idx] = self.encode_segmap(label)

    def _load_data(self, index):
        # image_path = os.path.join(self.root, self.files[index][:][1:])
        # image = np.asarray(Image.open(image_path), dtype=np.float32)
        # image_id = self.files[index].split("/")[-1].split(".")[0]
        # if self.use_weak_lables:
        #     label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        # else:
        #     label_path = os.path.join(self.root, self.labels[index][:][1:])
        # if label_path is not None:
        #     label = np.asarray(Image.open(label_path), dtype=np.int32)
        #     label = self.encode_segmap(np.array(label, dtype=np.uint8))
        #     return image, label, image_id
        return self.image_loaded[index], None#, image_id

    # @profile(precision=4, stream=open('/home/zhengyu/ours_video/memory/memory_pair_getitem_cycle.log', 'w+'))
    def __getitem__(self, index):

        image_path = os.path.join(self.root, self.files[index][:][1:])

        image = self.image_loaded[index]#np.asarray(Image.open(image_path))
        # if self.use_weak_lables:
        #     label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        # elif self.labels is not None:
        #     label_path = os.path.join(self.root, self.labels[index][:][1:])
        # else:
        label_path = None

        h, w, _ = image.shape

        longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
        image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
        if label_path is not None:
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            label = None

        crop_h, crop_w = self.crop_size, self.crop_size
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            if label is not None:
                label = cv2.copyMakeBorder(label, value=self.ignore_index, **pad_kwargs)

        x1 = random.randint(0, w + pad_w - crop_w)
        y1 = random.randint(0, h + pad_h - crop_h)

        max_iters = 50
        k = 0
        while k < max_iters:
            x2 = random.randint(0, w + pad_w - crop_w)
            y2 = random.randint(0, h + pad_h - crop_h)
            # crop relative coordinates should be a multiple of 8
            x2 = (x2 - x1) // self.stride * self.stride + x1
            y2 = (y2 - y1) // self.stride * self.stride + y1
            if x2 < 0: x2 += self.stride
            if y2 < 0: y2 += self.stride

            if (crop_w - abs(x2 - x1)) > 0 and (crop_h - abs(y2 - y1)) > 0:
                inter = (crop_w - abs(x2 - x1)) * (crop_h - abs(y2 - y1))
                union = 2 * crop_w * crop_h - inter
                iou = inter / union
                if iou >= self.iou_bound[0] and iou <= self.iou_bound[1]:
                    break
            k += 1

        if k == max_iters:
            x2 = x1
            y2 = y1
        # ul=upleft overlap左上角的点在该image中的坐标 br=bottomright  overlap右下角的点在该image中的坐标
        overlap1_ul = [max(0, y2 - y1), max(0, x2 - x1)]
        overlap1_br = [min(self.crop_size, self.crop_size + y2 - y1, h // self.stride * self.stride),
                       min(self.crop_size, self.crop_size + x2 - x1, w // self.stride * self.stride)]
        overlap2_ul = [max(0, y1 - y2), max(0, x1 - x2)]
        overlap2_br = [min(self.crop_size, self.crop_size + y1 - y2, h // self.stride * self.stride),
                       min(self.crop_size, self.crop_size + x1 - x2, w // self.stride * self.stride)]

        try:
            assert (overlap1_br[0] - overlap1_ul[0]) * (overlap1_br[1] - overlap1_ul[1]) == (
                        overlap2_br[0] - overlap2_ul[0]) * (overlap2_br[1] - overlap2_ul[1])
            assert overlap1_br[0] >= 0 and overlap1_ul[0] >= 0 and overlap1_br[1] >= 0 and overlap1_ul[1] >= 0
            assert overlap2_br[0] >= 0 and overlap2_ul[0] >= 0 and overlap2_br[1] >= 0 and overlap2_ul[1] >= 0
        except:
            print("k: {}".format(k))
            # print("h: {}, w: {}".format(h, w))
            print("image.shape: ", image.shape)
            print("x1: {}, x2: {}, y1: {}, y2: {}".format(x1, x2, y1, y2))
            print("image_path:", image_path)
            print("ul1: ", overlap1_ul)
            print("br1: ", overlap1_br)
            print("ul2: ", overlap2_ul)
            print("br2: ", overlap2_br)
            print("index: ", index)
            exit()

        image1 = image[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()
        image2 = image[y2:y2 + self.crop_size, x2:x2 + self.crop_size].copy()
        if label is not None:
            label1 = label[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()
            label2 = label[y2:y2 + self.crop_size, x2:x2 + self.crop_size].copy()

        try:
            assert image1[overlap1_ul[0]:overlap1_br[0], overlap1_ul[1]:overlap1_br[1]].shape == image2[overlap2_ul[0]:
                                                                                                        overlap2_br[0],
                                                                                                 overlap2_ul[1]:
                                                                                                 overlap2_br[1]].shape
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
            if label is not None:
                label1 = np.fliplr(label1)
            flip1 = True

        flip2 = False
        if random.random() < 0.5:
            image2 = np.fliplr(image2)
            if label is not None:
                label2 = np.fliplr(label2)
            flip2 = True
        flip = [flip1, flip2]

        image1 = self.train_transform(image1)
        image2 = self.train_transform(image2)

        images = torch.stack([image1, image2])
        if label is not None:
            labels = torch.from_numpy(np.stack([label1, label2]))
        else:
            labels = None
        return images, overlap1_ul, overlap1_br, overlap2_ul, overlap2_br, flip

    def decode_segmap(self, temp):
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

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

class ClipThermalVid(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
        self.STD = [[0.229, 0.224, 0.225], [0.229, 0.224, 0.225]]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')
        self.dataset = ClipThermalVidDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)

        super(ClipThermalVid, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                             dist_sampler=dist_sampler)
class ClipThermalVidDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 17
        self.void_classes = [0]
        self.valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(17)))

        self.datalist = kwargs.pop("datalist")
        self.stride = kwargs.pop('stride')
        self.iou_bound = kwargs.pop('iou_bound')
        self.hflip=kwargs.pop("hflip")
        self.clip_size = kwargs.pop("clip_size")
        self.jitter = kwargs["jitter"]
        self.scales=[0.08, 1]
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        super(ClipThermalVidDataset, self).__init__(**kwargs)
        # # self.palette = pallete.get_voc_pallete(self.num_classes)
       #
        # # clip_augmentations=
        # self.augmentations1 = get_composed_augmentations(clip1_augmentations)
        # self.augmentations2 = get_composed_augmentations(clip2_augmentations)
    def _set_files(self):
        prefix = "dataloaders/thermalour_splits{}".format(self.datalist)

        if self.split == "val":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split == "test":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split == "train_supervised":
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_{self.split}" + ".txt")
        elif self.split == "train_unsupervised":
            file_list = os.path.join(prefix, f"1-{self.n_unlabeled_ratio}_{self.split}" + ".txt")
        elif self.split == "train_vid":
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_train_supervised" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        if self.split == "train_unsupervised":
            file_list = [line.rstrip() for line in tuple(open(file_list, "r"))]
            self.files, self.labels = file_list, None
        else:
            file_list = [line.rstrip().split('	') for line in tuple(open(file_list, "r"))]
            self.files, self.labels = list(zip(*file_list))

        # file_list = [line.rstrip() for line in tuple(open(file_list, "r"))]
        self.image_loaded = np.empty([len(file_list), self.clip_size, self.base_size[0],self.base_size[1],3], dtype=float)
        # self.label_loaded = np.empty([len(file_list), self.base_size[0], self.base_size[1]], dtype=int)
        for idx in len(file_list):
            img_path = os.path.join(self.root, self.files[idx][:][1:])
            vid_info = img_path.split('/')[-1].split('.')[0].split('_')
            city, seq, cur_frame = vid_info[0], vid_info[1], vid_info[2]
            self.image_loaded[idx] = np.asarray(Image.open(img_path), dtype=np.float32)
            for fid in range(self.clip_size):
                neighbor_img_id = img_id + fid + 1
            self.image_loaded[idx] = np.asarray(Image.open(img_path), dtype=np.float32)
            # label_path = os.path.join(self.root, self.labels[idx][1][1:])
            # label = np.asarray(Image.open(label_path), dtype=np.uint8)
            # self.label_loaded[idx] = self.encode_segmap(label)

        clip_size = self.clip_size
        # self.interval=2
        img_id = int(cur_frame)
        nei_img = []
        nei_img.append(image)
        crop1_start, crop2_start, scale, overlap1_ul, overlap1_br, overlap2_ul, overlap2_br = self.get_crops(image)
        for fid in range(clip_size):
            neighbor_img_id = img_id + fid + 1
            neighbor_img_path = os.path.join(self.root, os.path.split(self.files[index][:][1:])[0],
                                             ("%s_%s_%06d.bmp" % (city, seq, neighbor_img_id)))
            neighbor_img = Image.open(neighbor_img_path)
            neighbor_img = np.array(neighbor_img, dtype=np.uint8)
            nei_img.append(neighbor_img)


    def _load_data(self, index):
        # image_path = os.path.join(self.root, self.files[index][1:])
        # image = np.asarray(Image.open(image_path), dtype=np.float32)
        # image_id = self.files[index].split("/")[-1].split(".")[0]
        # if self.use_weak_lables:
        #     label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        # else:
        #     label_path = os.path.join(self.root, self.labels[index][:][1:])
        # if label_path is not None:
        #     label = np.asarray(Image.open(label_path), dtype=np.int32)
        #     label = self.encode_segmap(np.array(label, dtype=np.uint8))
        #     return image, label, image_id
        return self.label_loaded[index], None#, image_id

    # @profile(precision=4, stream=open('/home/zhengyu/ours_video/memory/memory_clip_getitem.log', 'w+'))
    def __getitem__(self, index):

        image_path = self.label_loaded[index]#os.path.join(self.root, self.files[index][:][1:])
        vid_info = image_path.split('/')[-1].split('.')[0].split('_')
        city, seq, cur_frame = vid_info[0], vid_info[1], vid_info[2]
        image = np.asarray(Image.open(image_path))
        if self.labels is not None:
            label_path = os.path.join(self.root, self.labels[index][:][1:])
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            label = self.encode_segmap(np.array(label, dtype=np.uint8))
        else:
            label_path = None

        clip_size = self.clip_size
        # self.interval=2
        img_id = int(cur_frame)
        nei_img = []
        nei_img.append(image)
        crop1_start, crop2_start, scale, overlap1_ul, overlap1_br, overlap2_ul, overlap2_br = self.get_crops(image)
        for fid in range(clip_size):
            neighbor_img_id = img_id + fid +1
            neighbor_img_path = os.path.join(self.root, os.path.split(self.files[index][:][1:])[0],
                                             ("%s_%s_%06d.bmp" % (city, seq, neighbor_img_id)))
            neighbor_img = Image.open(neighbor_img_path)
            neighbor_img = np.array(neighbor_img, dtype=np.uint8)
            nei_img.append(neighbor_img)
        clip1_augmentations=dict()
        clip2_augmentations = dict()
        clip1_augmentations['rscrop'] = self.crop_size, crop1_start, scale
        clip2_augmentations['rscrop'] = self.crop_size, crop2_start, scale
        clip1_augmentations['hflip'] = self.hflip
        clip2_augmentations['hflip'] = self.hflip
        clip1_augmentations['jitter'] = self.jitter
        clip2_augmentations['jitter'] = self.jitter
        clip1_augmentations['colornorm'] = [self.MEAN, self.STD]
        clip2_augmentations['colornorm'] = [self.MEAN, self.STD]
        self.augmentations1 = get_composed_augmentations(clip1_augmentations)
        self.augmentations2 = get_composed_augmentations(clip2_augmentations)

        # if self.augment is True:
        #     nei_img_f, aug_f_p = self.augmentations1(nei_img)
        #     nei_img_b, aug_b_p = self.augmentations2(nei_img)
        nei_img_f, aug_f_p = self.augmentations1(nei_img.copy())
        input_f=torch.stack(nei_img_f)
        nei_img_b, aug_b_p = self.augmentations2(nei_img.copy())
        input_b = torch.stack(nei_img_b)

        # affine matrices
        def affine_sp_aug(aug_param):
        #affine_rscrop
            [x,y],boxw,boxh, outw, outh, hflip=aug_param[0][0], aug_param[0][1], aug_param[0][1], aug_param[0][2], aug_param[0][2], aug_param[1]
            theta_crop = torch.tensor([[outw/boxw, 0, outw/2-(outw/boxw)*(x+boxw/2)],
                                      [0, outh/boxh, outh/2-(outh/boxh)*(y+boxh/2)],
                                      [0, 0, 1]]).float()
            if hflip is False:
                theta_hflip = torch.tensor([[1, 0, 0],
                                           [0, 1, 0],
                                            [0, 0, 1]]).float()
            else:
                theta_hflip = torch.tensor([[-1, 0, outw],
                                            [0, 1, 0],
                                            [0, 0, 1]]).float()

            theta=torch.mm(theta_hflip,theta_crop)
            return theta

        theta_s=torch.tensor([[1/8, 0, 0],
                                [0, 1/8, 0],
                                [0, 0, 1]]).float() #frame到feature 320到40
        theta_f = affine_sp_aug(aug_f_p)
        theta_b = affine_sp_aug(aug_b_p)
        theta = theta_s@theta_b@theta_f.inverse()@theta_s.inverse()

        return input_f, input_b, theta[:-1]

    def get_crops(self, image):
        h, w, _ = image.shape
        max_iters = 50
        k = 0
        while k < max_iters:
            area = h*w
            scale = random.uniform(self.scales[0], self.scales[1])
            target_area = area*scale
            crop_w = int(round(math.sqrt(target_area)))
            crop_h = crop_w
            if 0 < crop_w <= w and 0 < crop_h <= h:
                break
            k += 1

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

        x1 = random.randint(0, w+pad_w-crop_w)
        y1 = random.randint(0, h+pad_h-crop_h)

        max_iters = 100
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
            print("ul1: ", overlap1_ul)
            print("br1: ", overlap1_br)
            print("ul2: ", overlap2_ul)
            print("br2: ", overlap2_br)
            exit()

        return [x1, y1], [x2, y2], scale, overlap1_ul, overlap1_br, overlap2_ul, overlap2_br

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