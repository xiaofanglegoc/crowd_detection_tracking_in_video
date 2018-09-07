from skimage.io import imread
import cv2
import numpy as np
from PIL import Image
#
import collections
caffe_root="/home/wang/Desktop/SSH/caffe-ssh/"
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, 'lib','gSLICrPy'))
sys.path.insert(0,caffe_root+'python')
from nms.nms_wrapper import nms
from gSLICrPy import __get_CUDA_gSLICr__, CUDA_gSLICr
from skimage.segmentation import mark_boundaries, find_boundaries
import matplotlib.pyplot as plt
from utils.blob import im_list_to_blob
from skimage.util import img_as_float

from utils.test_utils import _get_image_blob, _compute_scaling_factor, visusalize_detections
import caffe
#import feature_extraction import SSH
from utils.get_config import cfg_from_file, cfg, cfg_print
from SSH.test import detect
#  GOOGLENET ############3
#MODEL_DIR="/home/wang/Desktop/SSH/lib/gSLICrPy/googlenet"
#MODEL_DEF=os.path.join(MODEL_DIR,"bvlc_googlenet.prototxt")
#MODEL_WEIGHT=os.path.join(MODEL_DIR,"bvlc_googlenet.caffemodel")
#   SSH ##############
MODEL_DIR="/home/wang/Desktop/SSH/"
MODEL_DEF=os.path.join(MODEL_DIR,"SSH/models/test_ssh.prototxt")
MODEL_WEIGHT=os.path.join(MODEL_DIR,"output/scut/scut_train/SSH_iter_21000.caffemodel")
CFG_PATH=os.path.join(MODEL_DIR,"SSH/configs/shibuya.yml")
cfg_from_file(CFG_PATH)
cfg_print(cfg)

# Loading the network
cfg.GPU_ID = 0
caffe.set_mode_gpu()
caffe.set_device(0)

net  = caffe.Net(MODEL_DEF,MODEL_WEIGHT, caffe.TEST)
net.name = 'SSH_SCUT'
print('Done!')


pyramid = True if len(cfg.TEST.SCALES)>1 else False
OUT_PATH=os.path.join(MODEL_DIR,"results/gslic/detection/")
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
                    
COLORS=[(230,230,255),(128,128,255),(26,26,255),(255,255,0)]







def feature_SSH(net,im,branche=1):
    """
    :param net: the trained network
    :param im:  the image
    :return:feature 
    """
    #im_scale = _compute_scaling_factor(im.shape,50,200)
    # print("scale is ",im_scale)
    #blob = _get_image_blob(im,[im_scale])
    #blobs=[]
    #im_copy = im.astype(np.float32, copy=True) - [[[102.9801, 115.9465, 122.7717]]]
    #for scale in im_scale:
    #blobs.append({'data':im_list_to_blob([cv2.resize(im_copy, None, None, fx=im_scale, fy=im_scale,
#interpolation=cv2.INTER_LINEAR)])})

    # Adding im_info to the data blob
    #blob['im_info'] = np.array(
    #    [[blob['data'].shape[2], blob['data'].shape[3], im_scale]],
    #    dtype=np.float32)
    # create transformer for the input called 'data'
    #net.blobs['data'].reshape(1,        # batch size
    #                     3,         # 3-channel (BGR) images
    #                     100, 100)  # image size is 227x227
    #if im.shape[0] <100 or im.shape[1]<100:
    im=cv2.resize(im, (224,224),interpolation=cv2.INTER_LINEAR)
    print(im.shape)
    transformed_image = transformer.preprocess('data', im)
    im_blob = np.zeros((1, im.shape[0], im.shape[1], im.shape[2]),dtype=np.float32)
    im_blob[0, 0:im.shape[0], 0:im.shape[1], :] = im

    #im_blob=im_list_to_blob(transformed_image)
    net.blobs['data'].data[...] = im_blob
    # Reshape network input
    #net.blobs['data'].reshape(*(blob['data'].shape))
    #net.blobs['im_info'].reshape(*(blob['im_info'].shape))

    # Forward the network
    net_args = {'data': blob['data'].astype(np.float32, copy=False)}

    blobs_out = net.forward(**net_args)

    #Pool6_SSH_Feature=net.blob['pool6'].data
    branche_SSH_feature=net.blob['m{}@ssh_output'.format(branche)].data
    #M1_SSH_feature=net.blob['m1@ssh_output'].data
    #M2_SSH_feature=net.blob['m2@ssh_output'].data

    return branche_SSH_feature





def main():
    video = cv2.VideoCapture("/home/wang/Desktop/SSH/data/datasets/shibuya/shibuya1_1.mp4")
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()

    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    fragement_memory=3
    dictionary_cls_dets={}
    dictionary_frame={}
     
    count=0 
    ok, fist_image = video.read()
    image_level_cls_dets,_ = detect(net,fist_image,visualization_folder=OUT_PATH,visualize=True,pyramid=pyramid,video=True,plt_name=str(count))
    
    dictionary_cls_dets.update({str(count):image_level_cls_dets})
    dictionary_frame.update({str(count):fist_image})
    while True:

        # Read a new frame
        ok, image = video.read()
        curImage=image.copy()
        #if not str(count-1) in dictionary_frame.keys():

        if len(dictionary_frame)<fragement_memory:
            dictionary_frame.update({str(count):curImage})
            #dictionary_frame=collections.OrderedDict(sorted(dictionary_frame.items()))
        #else:
        #    dictionary_frame.pop(str(count-fragement_memory))
        #    dictionary_frame.update({str(count):curImage})


        #image = imread('./example.jpg')
        img_f=img_as_float(image)

        img_size_y, img_size_x = image.shape[0:2]
        image = image[:, :, ::-1].flatten().astype('uint8')

        __CUDA_gSLICr__ = __get_CUDA_gSLICr__()


        out_name=os.path.join(BASE_DIR,'results','gslic','shibuya1_1/'+str(count))
        CUDA_gSLICr(__CUDA_gSLICr__,
                    image,
                    img_size_x=img_size_x,
                    img_size_y=img_size_y,
                    n_segs=10,
                    spixel_size=20,
                    coh_weight=0.6,
                    n_iters=50,
                    color_space=2,
                    segment_color_space=2,
                    segment_by_size=True,
                    enforce_connectivity=True,
                    out_name=out_name)
        f=open(out_name+'.txt','rb')
        #sup_map=np.zeros((1,img_size_x*img_size_y))
        supperpixel_list=[]
        c=0
        for l in f.readlines():
            supperpixel_list.append(int(l)+1)
            c+=1
        superpixel_map=np.array(supperpixel_list)
        superpixel_map=np.reshape(superpixel_map,(img_size_y,img_size_x))
        # Display result
        #superpixel_boundary=mark_boundaries(img_f,superpixel_map) 
        #cv2.imshow("boundary superpixel",superpixel_boundary)
        #cv2.imshow("supixel ",mark_boundaries(img_f,superpixel_map.astype(int)))
        image_level_cls_dets,_ = detect(net,curImage,visualization_folder=OUT_PATH,visualize=True,pyramid=pyramid,video=True,plt_name=str(count))

        if len(dictionary_cls_dets)<fragement_memory:
            dictionary_cls_dets.update({str(count):image_level_cls_dets})
        if count>=fragement_memory:
            # we do frame subtraction
            diff=np.zeros(curImage.shape).astype('uint8')
            for key, im in dictionary_frame.items():
                print('current is {} abs substracting {} '.format(count,key))
                diff=cv2.absdiff(curImage,im)+diff;
            
            # threshold the diff
            thresh = cv2.threshold(cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)[1]
            set_motion_superpixel=np.unique(superpixel_map[thresh==255])
            rects1=[]
            show_on_currImage1=curImage.copy()
            Image_curImage=Image.fromarray(curImage)
            #batch_motion_superpixel_feature_M1={}

            #image_level_cls_dets,_ = detect(net,curImage,visualization_folder=OUT_PATH,visualize=True,pyramid=pyramid,video=True,plt_name=str(count))
            for key,bbox  in dictionary_cls_dets.items():
                
                print("add {} bbox on {} current".format(key,count))
                for b in bbox:
                    cv2.rectangle(show_on_currImage1,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(255,255,0),1)
            for b in image_level_cls_dets:
                cv2.rectangle(show_on_currImage1,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(255,0,0),2,cv2.LINE_AA)
            """
            for s in set_motion_superpixel:
                if s==0:
                    continue

                binary_motion_superpixel=np.zeros((img_size_y,img_size_x))
                binary_motion_superpixel[superpixel_map==s]=255

                _,cnts,hierarchy=cv2.findContours(binary_motion_superpixel.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    peri=cv2.arcLength(c,True)
                    approx=cv2.approxPolyDP(c,0.02*peri,True)
                    x,y,w,h=cv2.boundingRect(approx)
                    if h>5 and w>5:
                        rect=(x,y,w,h)

                        #imCrop= Image_curImage.crop((x,y,x+w,y+h))
                        #cv2.imshow("path",np.asarray(imCrop))
                        #feature=feature_SSH(net,imCrop,branche=1)
                        #batch_motion_superpixel_feature_M1.update({str(s):feature})
                        #cls_dets,_ = detect(net,np.asarray(imCrop),visualization_folder=OUT_PATH,visualize=True,pyramid=pyramid,video=True,plt_name=str(count)+'_'+str(s))
                        rects1.append(rect)
                        cv2.rectangle(show_on_currImage1,(x,y),(x+w,y+h),(255,0,0),1)
            cv2.imshow("motion superpixel",show_on_currImage1)
            """
            #cv2.imshow("bounding box",binary_motion_superpixel)
            _,cnts,hierarchy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            rects=[]
            #show_on_currImage=curImage.copy()
            #superpixel_map_cv=cv2.CreateMat(img_size_y,img_size_x, cv2.CV_32FC3)
            for c in cnts:
                peri=cv2.arcLength(c,True)
                approx=cv2.approxPolyDP(c,0.02*peri,True)
                x,y,w,h=cv2.boundingRect(approx)
                if h>2 and w>2:
                    rect=(x,y,w,h)
                    rects.append(rect)
                    cv2.rectangle(show_on_currImage1,(x,y),(x+w,y+h),(255,0,255),1)
            cv2.imshow("boundary superpixel",show_on_currImage1)
        
            #if count%3==0:
            #    fist_image=curImage.copy()
            #    prediff=diff.copy()
            dictionary_frame.pop(str(count-fragement_memory))
            dictionary_frame.update({str(count):curImage})
            dictionary_cls_dets.pop(str(count-fragement_memory))
            dictionary_cls_dets.update({str(count):image_level_cls_dets})
        count+=1 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break  
if __name__ == '__main__':
    main()
