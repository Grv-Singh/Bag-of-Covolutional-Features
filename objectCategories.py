import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm

TRAIN_DIR = "D:/Major_Code/train"
TEST_DIR = "D:/Major_Code/test"
IMG_SIZE = 100
LR = 1e-3

MODEL_NAME = 'objectCategories-{}-{}.model'.format(LR, '2conv-basic')
#MODEL_NAME = 'objectCategories-{}-{}.model'.format(LR, 'conv5') 
#MODEL_NAME = 'objectCategories-{}-{}.model'.format(LR, 'inception 4e') 
#MODEL_NAME = 'objectCategories-{}-{}.model'.format(LR, 'conv5-3')

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
		elif word_label == 'mediumresidential': return [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
		elif word_label == 'mobilehomepark': return 	[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
		elif word_label == 'overpass': return 			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
		elif word_label == 'parkinglot': return 		[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
		elif word_label == 'river': return 				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
		elif word_label == 'runway': return 			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
		elif word_label == 'sparseresidential': return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
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
        
    #shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()

#train_data= np.load('train_data.npy')

import tflearn
from tflearn.layers.conv import conv_2d, avg_pool_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = avg_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 21, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-20]
test = train_data[-20:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

#training here
model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=10, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

import matplotlib.pyplot as plt

test_data = process_test_data()

#etest_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
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
else: str_label='tenniscourt'

y.imshow(orig,cmap='gray')
plt.title(str_label)
y.axes.get_xaxis().set_visible(False)
y.axes.get_yaxis().set_visible(False)
plt.show()

