from skimage.io import imread
import numpy as np
from gSLICrPy import __get_CUDA_gSLICr__, CUDA_gSLICr
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.util import img_as_float
def main():
    image = imread('./example.jpg')
    img_f=img_as_float(image)

    img_size_y, img_size_x = image.shape[0:2]
    image = image[:, :, ::-1].flatten().astype('uint8')

    __CUDA_gSLICr__ = __get_CUDA_gSLICr__()

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
                out_name='example_results')
    f=open("seg_example_results.txt",'rb')
    #sup_map=np.zeros((1,img_size_x*img_size_y))
    supperpixel_list=[]
    count=0
    for l in f.readlines():
        supperpixel_list.append(l)
        count+=1
    print("image size is ",img_size_x*img_size_y)
    print("text file length is ",count)
    superpixel_map=np.array(supperpixel_list)
    superpixel_map=np.reshape(superpixel_map,(img_size_y,img_size_x))
    print("superpxiel map size is ",superpixel_map.shape,img_f.shape)
    fig = plt.figure("Superpixels" )
    ax=fig.add_subplot(1,1,1,)
    ax.imshow(mark_boundaries(img_f,superpixel_map.astype(int)))
    plt.axis("off")
    plt.show()
if __name__ == '__main__':
    main()
