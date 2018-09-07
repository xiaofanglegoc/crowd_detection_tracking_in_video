from skimage.io import imread
import cv2
import numpy as np
#from nms.nms_wrapper import nms
from gSLICrPy import __get_CUDA_gSLICr__, CUDA_gSLICr
from skimage.segmentation import mark_boundaries, find_boundaries
import matplotlib.pyplot as plt

from skimage.util import img_as_float


caffe_root="/home/wang/Desktop/SSH/caffe-ssh/"
import sys
sys.path.insert(0,caffe_root+'python')
from utils.test_utils import _get_image_blob, _compute_scaling_factor, visusalize_detections
import caffe

import os
#  GOOGLENET ############3
#MODEL_DIR="/home/wang/Desktop/SSH/lib/gSLICrPy/googlenet"
#MODEL_DEF=os.path.join(MODEL_DIR,"bvlc_googlenet.prototxt")
#MODEL_WEIGHT=os.path.join(MODEL_DIR,"bvlc_googlenet.caffemodel")
#   SSH ##############
MODEL_DIR="/home/wang/Desktop/SSH/"
MODEL_DEF=os.path.join(MODEL_DIR,"SSH","model","test_ssh.prototxt" )
MODEL_WEIGHT=os.path.join(MODEL_DIR,"output/scut/scut_train/SSH_iter_21000.caffemodel")


caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(MODEL_DEF,      # defines the structure of the model
                MODEL_WEIGHT,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
#mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)
# create transformer for the input called 'data'
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
#transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
# load ImageNet labels
#labels_file = os.path.join(MODEL_DIR, "synset_words.txt")
#labels = np.loadtxt(labels_file, str, delimiter='\t')

def main():
    video = cv2.VideoCapture("/home/wang/Desktop/SSH/data/datasets/shibuya/shibuya1_1.mp4")
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    # just try
    #net.blobs['data'].data[...] = transformer.preprocess('data', frame)
        # perform classification
    #net.forward()

    # obtain the output probabilities
    #output_prob = net.blobs['prob'].data[0]

    # sort top five predictions from softmax output
    #top_inds = output_prob.argsort()[::-1][:5]
    #print(output_prob.argmax())
    #print 'output label:', labels[output_prob.argmax()]

    if not ok:
        print('Cannot read video file')
        sys.exit()
     
    count=0 
    ok, fist_image = video.read()
    while True:
        # Read a new frame
        ok, image = video.read()
        curImage=image.copy()
        #image = imread('./example.jpg')
        img_f=img_as_float(image)

        img_size_y, img_size_x = image.shape[0:2]
        image = image[:, :, ::-1].flatten().astype('uint8')

        __CUDA_gSLICr__ = __get_CUDA_gSLICr__()


        out_name='shibuya1_1/'+str(count)
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
        #print('lable is ',set(supperpixel_list))
        #print("image size is ",img_size_x*img_size_y)
        #print("text file length is ",c)
        superpixel_map=np.array(supperpixel_list)
        superpixel_map=np.reshape(superpixel_map,(img_size_y,img_size_x))
        # Display result
        #superpixel_boundary=mark_boundaries(img_f,superpixel_map) 
        #cv2.imshow("boundary superpixel",superpixel_boundary)
        #cv2.imshow("supixel ",mark_boundaries(img_f,superpixel_map.astype(int)))
        prediff=cv2.absdiff(curImage,fist_image)
        if count>3:
            # we do frame subtraction
            
            diff=cv2.absdiff(curImage,fist_image);
            print(type(diff))
            diff1=cv2.absdiff(curImage,preImage);
            diff=diff+diff1+prediff
            #print("diff",diff.shape)
            #cv2.imshow("diff",diff)
            # threshold the diff
            thresh = cv2.threshold(cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)[1]
            #cv2.imshow("threshold diff",thresh)
            #print("thresh diff",thresh.shape)
            # find motion superpixels
            #print(np.unique(thresh))
            
            set_motion_superpixel=np.unique(superpixel_map[thresh==255])
            #print(set_motion_superpixel)
            #index_move_superpixel=np.multiply(superpixel_map, thresh)
            # imshow these motions superpixels  to check 
            #print(len(np.unique(index_move_superpixel)))
            
            #boundary_motion_superpixel=np.zeros((img_size_y,img_size_x))
            #index_set=np.unique(index_move_superpixel)
            #print(np.sort(index_set))
            rects1=[]
            show_on_currImage1=curImage.copy()


            for s in set_motion_superpixel:
                if s==0:
                    continue

                binary_motion_superpixel=np.zeros((img_size_y,img_size_x))
                binary_motion_superpixel[superpixel_map==s]=255

                #thresh_binary = cv2.threshold(binary_motion_superpixel, 0, 255, cv2.THRESH_BINARY)[1]
                #print(thresh_binary.dtype, thresh_binary.shape)
                _,cnts,hierarchy=cv2.findContours(binary_motion_superpixel.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    peri=cv2.arcLength(c,True)
                    approx=cv2.approxPolyDP(c,0.02*peri,True)
                    x,y,w,h=cv2.boundingRect(approx)
                    if h>2 and w>2:
                        rect=(x,y,w,h)
                        rects1.append(rect)
                        cv2.rectangle(show_on_currImage1,(x,y),(x+w,y+h),(255,0,0),1)
            cv2.imshow("motion superpixel",show_on_currImage1)
                
            #cv2.imshow("bounding box",binary_motion_superpixel)
            _,cnts,hierarchy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            rects=[]
            show_on_currImage=curImage.copy()
            #superpixel_map_cv=cv2.CreateMat(img_size_y,img_size_x, cv2.CV_32FC3)
            for c in cnts:
                peri=cv2.arcLength(c,True)
                approx=cv2.approxPolyDP(c,0.02*peri,True)
                x,y,w,h=cv2.boundingRect(approx)
                if h>2 and w>2:
                    rect=(x,y,w,h)
                    rects.append(rect)
                    cv2.rectangle(show_on_currImage,(x,y),(x+w,y+h),(255,0,0),1)
            cv2.imshow("boundary superpixel",show_on_currImage)
            if count%3==0:
                fist_image=curImage.copy()
                prediff=diff.copy()
        preImage=curImage.copy()
        count+=1 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break  
        #print("superpxiel map size is ",superpixel_map.shape,img_f.shape)
        #fig = plt.figure("Superpixels" )
        # ax=fig.add_subplot(1,1,1,)
        #ax.imshow(mark_boundaries(img_f,superpixel_map.astype(int)))
        #plt.axis("off")
        #plt.show()
if __name__ == '__main__':
    main()
