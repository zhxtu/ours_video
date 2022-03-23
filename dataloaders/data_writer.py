import os
from glob import glob
import random
# import numpy as np
image_root= '/hdd2/zy/Dataset/ThermalData/'
train_image_list=glob(os.path.join(image_root, 'Img8bit', 'train', '*', '*.bmp'))

valinfo=open('/hdd2/zy/Dataset/ThermalData/val.txt','w')
val_image_list=glob(os.path.join(image_root, 'Img8bit', 'val', '*', '*.bmp'))
for i in range(len(val_image_list)):
    imgpath=val_image_list[i]
    imgpath='/'+imgpath[imgpath.index('Img8bit'):]
    vid=imgpath.split('/')[-2]
    name=imgpath.split('/')[-1]
    gtname=name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2]+'_gtLabelIds.png'
    gtpath=os.path.join('/gtFine', 'val', vid, gtname)
    valinfo.write(imgpath)
    valinfo.write('\t')
    valinfo.write(gtpath)
    valinfo.write('\r\n')

testinfo=open('/hdd2/zy/Dataset/ThermalData/test.txt','w')
test_image_list=glob(os.path.join(image_root, 'Img8bit', 'test', '*', '*.bmp'))
for i in range(len(test_image_list)):
    imgpath=test_image_list[i]
    imgpath = '/' + imgpath[imgpath.index('Img8bit'):]
    vid=imgpath.split('/')[-2]
    name=imgpath.split('/')[-1]
    gtname=name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2]+'_gtLabelIds.png'
    gtpath=os.path.join('/gtFine', 'test', vid, gtname)
    testinfo.write(imgpath)
    testinfo.write('\t')
    testinfo.write(gtpath)
    testinfo.write('\r\n')

numofimg=len(train_image_list)
idx=(random.sample(range(0,numofimg),int(numofimg/2)))
idx.sort()
l = open('/hdd2/zy/Dataset/ThermalData/subset_train/train_aug_labeled_1-2.txt','w')
un = list(range(0,numofimg))

for i in range(len(idx)):
    imgpath=train_image_list[idx[i]]
    imgpath = '/' + imgpath[imgpath.index('Img8bit'):]
    vid=imgpath.split('/')[-2]
    name=imgpath.split('/')[-1]
    gtname=name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2]+'_gtLabelIds.png'
    gtpath=os.path.join('/gtFine', 'train', vid, gtname)
    l.write(imgpath)
    l.write('\t')
    l.write(gtpath)
    l.write('\r\n')
    un.remove(idx[i])

u=open('/hdd2/zy/Dataset/ThermalData/subset_train/train_aug_unlabeled_1-2.txt','w')

for i in range(len(un)):
    imgpath = train_image_list[un[i]]
    imgpath = '/' + imgpath[imgpath.index('Img8bit'):]
    vid = imgpath.split('/')[-2]
    name = imgpath.split('/')[-1]
    gtname = name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2] + '_gtLabelIds.png'
    gtpath = os.path.join('/gtFine', 'train', vid, gtname)
    u.write(imgpath)
    u.write('\t')
    u.write(gtpath)
    u.write('\r\n')