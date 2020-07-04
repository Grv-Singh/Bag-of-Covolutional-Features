from PIL import Image
import numpy as np
from grey import GWO
import os

def threshold(t, image, filename):
    image_tmp = np.asarray(image)
    intensity_array = list(np.where(image_tmp<t, 0, 255).reshape(-1))
    image.putdata(intensity_array)
    image.save(filename[:filename.find('.tif')]+'x'+filename[filename.find('.tif'):])

def k(n):
	n = ((2-len(str(n)))*'0') + n
	return n

def main():
    print("\n-----------------------------------------Test Data Optimization---------------------------------------------\n")
    cls=['agricultural','buildings','freeway','mediumresidential','river','tenniscourt']
    for a in range(6):
        print('\n'+cls[a]+'\n')
        for i in range(1,28):
            filename = 'C:/BOVW_CNN_GWO_w_Dataset/Datasets/BOVW/image_regular/test/'+cls[a]+'/'+cls[a]+k(str(i))+'.tif'
            im = Image.open(filename)
            im.load()
            im_gray = im.convert('L') # translate to  gray map
            gwo = GWO(im_gray)
            gwo.hunt()
            threshold_arr = gwo.result_curve()
            best_threshold = threshold_arr[0]
            #print (threshold_arr)
            print (best_threshold)
            threshold(best_threshold, im_gray, filename)
            
    print("\n\n-----------------------------------------Train Data Optimization---------------------------------------------\n")
    for a in range(6):
        print('\n'+cls[a]+'\n')
        for i in range(1,66):
            filename = 'C:/BOVW_CNN_GWO_w_Dataset/Datasets/BOVW/image_regular/train/'+cls[a]+'/'+cls[a]+k(str(i))+'.tif'
            im = Image.open(filename)
            im.load()
            im_gray = im.convert('L') # translate to  gray map
            gwo = GWO(im_gray)
            gwo.hunt()
            threshold_arr = gwo.result_curve()
            best_threshold = threshold_arr[0]
            #print (threshold_arr)
            print (best_threshold)
            threshold(best_threshold, im_gray, filename)

if __name__ == "__main__":
    main()