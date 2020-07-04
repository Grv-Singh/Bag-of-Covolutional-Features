#OpenCV Library
import cv2                 
import numpy as np         
import os                  
from random import shuffle
#Library to maintain files and directories 
from tqdm import tqdm
import math

#Datasets
TRAIN_DIR = "C:\BOVW_CNN_GWO_w_Dataset\Datasets\CNN\\train_cnn"
TEST_DIR = "C:\BOVW_CNN_GWO_w_Dataset\Datasets\CNN\\test_cnn"

#Image of size 100px X 100px
IMG_SIZE = 100

#Learning Rate of 0.01
LR = 1e-3

#2conv model
MODEL_NAME = 'objectCategories-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
		word_label = img.split('.')[0]
		if word_label == 'agricultural': return 		[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'airplane': return 			[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'baseballdiamond': return 	[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'beach': return 				[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'buildings': return 			[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'chaparral': return 			[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'denseresidential': return 	[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'forest': return 			[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'freeway': return 			[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'golfcourse': return 		[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'harbor': return 			[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
		elif word_label == 'intersection': return 		[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
		elif word_label == 'mediumresidential': return  [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
		elif word_label == 'mobilehomepark': return 	[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
		elif word_label == 'overpass': return 			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
		elif word_label == 'parkinglot': return 		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
		elif word_label == 'river': return 				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
		elif word_label == 'runway': return 			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
		elif word_label == 'sparseresidential': return  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
		elif word_label == 'storagetanks': return 		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
		elif word_label == 'tenniscourt': return 		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    #saved form of train model
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])   
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

print("\n\t\t\t\t-----------Feature Extraction using Convolutional Neural Network-----------\n\n")

train_data = create_train_data()

#Performing CNN

import tflearn
from tflearn.layers.conv import conv_2d, avg_pool_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

#Resetting plot
tf.reset_default_graph()

#Input Layer
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

#1st Layer set
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

#2nd Layer set
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

#3rd Layer set
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

#4th Layer set
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

#5th Layer set
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

#6th Layer set
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

#Output Layer
convnet = fully_connected(convnet, 21, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

numm = int(input("\n How many images do you want to use? \n"))
print("\nOK!\n")

#Using top images to test and train
train = train_data[:-numm]
test = train_data[-numm:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=10, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

import matplotlib.pyplot as plt

test_data = process_test_data()

fig=plt.figure()

for num,data in enumerate(test_data[:numm]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(math.sqrt(numm),math.sqrt(numm),num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0: str_label='agricultural'
    elif  np.argmax(model_out) == 1: str_label='airplane'
    elif np.argmax(model_out) == 2: str_label='baseballdiamond'
    elif np.argmax(model_out) == 3: str_label='beach'
    elif  np.argmax(model_out) == 4: str_label='buildings'
    elif np.argmax(model_out) == 5: str_label='chaparral'
    elif np.argmax(model_out) == 6: str_label='denseresidential'
    elif  np.argmax(model_out) == 7: str_label='forest'
    elif np.argmax(model_out) == 8: str_label='freeway'
    elif np.argmax(model_out) == 9: str_label='golf Course'
    elif  np.argmax(model_out) == 10: str_label='harbor'
    elif np.argmax(model_out) == 11: str_label='intersection'
    elif np.argmax(model_out) == 12: str_label='mediumresidential'
    elif  np.argmax(model_out) == 13: str_label='mobilehomepark'
    elif np.argmax(model_out) == 14: str_label='overpass'
    elif np.argmax(model_out) == 15: str_label='parkinglot'
    elif  np.argmax(model_out) == 16: str_label='river'
    elif np.argmax(model_out) == 17: str_label='runway'
    elif np.argmax(model_out) == 18: str_label='sparseresidential'
    elif  np.argmax(model_out) == 19: str_label='storagetanks'
    else: str_label='airplane'
    
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

with open('output_model.csv','w') as f:
    f.write('ID,Label\n')
            
with open('ouut_model.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))