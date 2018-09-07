import cv2

img=cv2.imread("example.jpg");
f=open("seg_example_results.txt",'rb')
w,h,_=img.shape
count=0
for l in f.readline():
    count+=1
print w*h, count
