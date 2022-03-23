clc,
clear all;
img=imread('C:\Users\zhxtu\Desktop\panda.jpg');
[h,w,c]=size(img);
box=[50, 20,100, 100 ];
x=50;y=20;boxw=100;boxh=100;
outw=200;
outh=200;
xform=[outw/boxw 0 outw/2-(outw/boxw)*(x+boxw/2);0, outh/boxh, outh/2-(outh/boxh)*(y+boxh/2);0, 0, 1]';
tform_translate = maketform('affine', xform);
[img_trans xdata ydata]= imtransform(img, tform_translate);