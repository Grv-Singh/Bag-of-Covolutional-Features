from PIL import Image
import numpy as np
from genetic import GA
from grey import GWO

def threshold(t, image, filename):
    image_tmp = np.asarray(image)
    intensity_array = list(np.where(image_tmp<t, 0, 255).reshape(-1))
    image.putdata(intensity_array)
    image.show()
    image.save(filename)

def k(n):
	n = ((3-len(str(n)))*'0') + n
	return n

def main():
	for i in range(100):
		filename = 'images/'+k(str(i))+'.jpeg'
		im = Image.open(filename)
    	im.load()
    	im.show()
    	im_gray = im.convert('L') # translate to  gray map

    	ga = GA(im_gray)
    	for x in range(50):
        	ga.evolve()
    	best_threshold = ga.result()
    	print (best_threshold)
    	gwo = GWO(im_gray)
    	gwo.hunt()
    	threshold_arr = gwo.result_curve()
    	best_threshold = threshold_arr[0]
    	print (threshold_arr)
    	print (best_threshold)

    	threshold(best_threshold, im_gray, filename)

if __name__ == "__main__":
    main()