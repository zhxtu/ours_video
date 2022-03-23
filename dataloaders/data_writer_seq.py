import os
from glob import glob
import random
# import numpy as np
image_root= '/hdd2/zy/Dataset/ThermalData/'
train_image_list=glob(os.path.join(image_root, 'Img8bit', 'train', '*', '*.bmp'))

# valinfo=open('/hdd2/zy/Dataset/ThermalData/val.txt','w')
# val_image_list=glob(os.path.join(image_root, 'Img8bit', 'val', '*', '*.bmp'))
# for i in range(len(val_image_list)):
#     imgpath=val_image_list[i]
#     imgpath='/'+imgpath[imgpath.index('Img8bit'):]
#     vid=imgpath.split('/')[-2]
#     name=imgpath.split('/')[-1]
#     gtname=name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2]+'_gtLabelIds.png'
#     gtpath=os.path.join('/gtFine', 'val', vid, gtname)
#     valinfo.write(imgpath)
#     valinfo.write('\t')
#     valinfo.write(gtpath)
#     valinfo.write('\r\n')
#
# testinfo=open('/hdd2/zy/Dataset/ThermalData/test.txt','w')
# test_image_list=glob(os.path.join(image_root, 'Img8bit', 'test', '*', '*.bmp'))
# for i in range(len(test_image_list)):
#     imgpath=test_image_list[i]
#     imgpath = '/' + imgpath[imgpath.index('Img8bit'):]
#     vid=imgpath.split('/')[-2]
#     name=imgpath.split('/')[-1]
#     gtname=name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2]+'_gtLabelIds.png'
#     gtpath=os.path.join('/gtFine', 'test', vid, gtname)
#     testinfo.write(imgpath)
#     testinfo.write('\t')
#     testinfo.write(gtpath)
#     testinfo.write('\r\n')


subnum=1
l = open('./thermalseq_splits1/1374_train_unsupervised.txt','w')
u = open('./thermalseq_splits1/1-{}_train_unsupervised.txt'.format(2*subnum),'w')
for i in range(len(train_image_list)):
    imgpath = train_image_list[i]
    imgpath = '/' + imgpath[imgpath.index('Img8bit'):]
    vid = imgpath.split('/')[-2]
    vid_info = imgpath.split('/')[-1].split('_')
    city, seq, cur_frame = vid_info[0], vid_info[1], vid_info[2]
    f4_id = int(cur_frame)
    f4_path = os.path.join('/IR_sequence', 'train', 'DJI_' + seq, ("%s_%s_%06d.bmp" % (city, seq, f4_id)))
    # f4_path = os.path.join('/IR_sequence', 'train', 'DJI_' + seq, ("%s_%s_%06d_IRframes.bmp" % (city, seq, f4_id)))
    gt_path=os.path.join('/gtFine', 'train', 'DJI_' + seq, ("%s_%s_%06d_gtLabelIds.png" % (city, seq, f4_id)))
    l.write(f4_path)
    l.write('\t')
    l.write(gt_path)
    l.write('\r\n')

    for j in range(subnum):
        f3_id = f4_id - (j+1)*random.randint(1, 2)
        f5_id = f4_id + (j+1)*random.randint(1, 2)
        f3_path = os.path.join('/IR_sequence', 'train', 'DJI_' + seq, ("%s_%s_%06d.bmp" % (city, seq, f3_id)))
        f5_path = os.path.join('/IR_sequence', 'train', 'DJI_' + seq, ("%s_%s_%06d.bmp" % (city, seq, f5_id)))
        # f3_path = os.path.join('/IR_sequence', 'train', 'DJI_' + seq, ("%s_%s_%06d_IRframes.bmp" % (city, seq, f3_id)))
        # f5_path = os.path.join('/IR_sequence', 'train', 'DJI_' + seq, ("%s_%s_%06d_IRframes.bmp" % (city, seq, f5_id)))
        u.write(f3_path)
        u.write('\r\n')
        u.write(f5_path)
        u.write('\r\n')