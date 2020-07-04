import cv2
import numpy as np 
from glob import glob 
import argparse
from helpers import *
from matplotlib import pyplot as plt 
import os

class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel(self):
        
       # Reading files and preparing file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)

        # Extracting SIFT Features from each image
        label_count = 0       
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print("\n Computing Features for ", word, "\n")
            for im in imlist:
                cv2.imshow("im", im)
                cv2.waitKey(10)
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                self.descriptor_list.append(des)
            label_count += 1

        # Performing clustering
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)

        # Showing vocabulary trained
        self.bov_helper.plotHist()
        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)


    def recognize(self,test_img, test_image_path=None):

        kp, des = self.im_helper.features(test_img)
        print("\n",kp,"\n")
        print("\n",des.shape,"\n")

        # Generating vocabulary for test image
        vocab = np.array( [[ 0 for i in range(self.no_clusters)]])

        # locate nearest clusters for each of 
        # the visual word (feature) present in the image
        
        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        print("\n",test_ret,"\n")

        print("\n",vocab,"\n")
        for each in test_ret:
            vocab[0][each] += 1

        print("\n",vocab,"\n")
        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)

        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        print("\n Image belongs to class : ", self.name_dict[str(int(lb[0]))], "\n")
        
        
        
        return lb



    def testModel(self):
        """ 
        This method is to test the trained classifier
        read all images from testing path 
        use BOVHelpers.predict() function to obtain classes of each image
        """
        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        predictions = []

        for word, imlist in self.testImages.items():
            print("\n processing " ,word, "\n")
            for im in imlist:
                print("\n",imlist[0].shape,"\n", imlist[1].shape,"\n")
                print("\n",im.shape,"\n")
                cl = self.recognize(im)
                print("\n",cl,"\n")
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))]
                    })

        print("\n",predictions,"\n")
        for each in predictions:
            cv2.imshow(each['object_name'], each['image'])
            cv2.waitKey(10)
            cv2.destroyWindow(each['object_name'])
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()

    def print_vars(self):
        pass

if __name__ == '__main__':
    # parse cmd args
    parser = argparse.ArgumentParser(
            description=" Bag of visual words Major Project"
        )
    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)

    args =  vars(parser.parse_args())
    print(args)
    
    bov = BOV(no_clusters=10)

    # set training paths
    bov.train_path = args['train_path'] 

    # set testing paths
    bov.test_path = args['test_path'] 

    # train the model
    bov.trainModel()

    # test model
    bov.testModel()