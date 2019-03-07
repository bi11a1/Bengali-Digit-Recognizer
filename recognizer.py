import cv2
import numpy as np
import csv
import os
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

IMG_DIR = 'image_a'
LABEL_DIR = 'label_a'
TEST_DIR = 'test'
CREATE_DATA = 0
IMG_SIZE = 32
PERFORM_TRAINING = 0
NO_OF_CLASSES = 10
LR = 0.001
N_EPOCH = 10
MODEL_NAME = 'bengali_digit_recognizer.model'

# --------------------------------------------------------------------------------------------------------
def preprocess(img):
	# Converting into binary
	img = cv2.imread(img, 0)
	_, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# Swap color to make white pixel as foreground
	thresh_img = cv2.bitwise_not(thresh_img)

	# Separating region of interest and resizing binary image
	mask = thresh_img > 0
	cropped = thresh_img[np.ix_(mask.any(1),mask.any(0))]
	cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

	return cropped

# --------------------------------------------------------------------------------------------------------
def create_labeled_data():
	labeled_data = []

	# Reading csv file
	file = open(LABEL_DIR+"/label.csv", "r")
	reader = csv.reader(file)

	# Coloumn 0 contains the image name
	# Coloumn 3 contains the image label
	x = 0
	for coloumn in reader:
		x = x + 1

		# First row contains heading in csv file
		if(x == 1):
			continue

		img = IMG_DIR + '/' + coloumn[0]
		label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		label[int(coloumn[3])] = 1
		preprocessed_img = preprocess(img)
		labeled_data.append([np.array(preprocessed_img), np.array(label)])
		print(x)
	np.save('data.npy', labeled_data)

# --------------------------------------------------------------------------------------------------------
print('---Creating dataset---')
if(CREATE_DATA):
	create_labeled_data()
data = np.load('data.npy')

# --------------------------------------------------------------------------------------------------------
print('---Separating training and testing data---')

# Taking 20% of the total data for validation
validation_size = int(len(data)*.2)

train_data = data[:-validation_size]
test_data = data[-validation_size:]

# --------------------------------------------------------------------------------------------------------
print('---Creating network model---')
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, NO_OF_CLASSES, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='logs')

# --------------------------------------------------------------------------------------------------------
if(PERFORM_TRAINING):
    print('---Training the model---')
    X = np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train_data]
    test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test_data]
    model.fit({'input': X}, {'targets': Y}, n_epoch=N_EPOCH, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=1000, show_metric=True, batch_size=128, run_id=MODEL_NAME)
    model.save('model/'+MODEL_NAME)

# --------------------------------------------------------------------------------------------------------
if os.path.exists('model/{}.meta'.format(MODEL_NAME)):
    print('---Loading trained model---')
    model.load('model/'+MODEL_NAME)

    error_count = 0
    for num, data in enumerate(test_data):
	    img_data = data[0]
	    img_data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
	    model_out = model.predict([img_data])[0]
	    str_label = np.argmax(model_out)
	    actual = np.argmax(data[1])
	    if(actual != str_label):
	        error_count += 1
    print('Found', error_count, 'errors out of', len(test_data), 'images')
    print('Accuracy on test data: %.2f %%' % ((1-(error_count/len(test_data)))*100))

else:
    print('Model not found')

# --------------------------------------------------------------------------------------------------------
print('---Checking unseen test---')
fig = plt.figure()
fig.canvas.set_window_title('Bengali Digit Recognizer')

new_test = os.listdir(TEST_DIR)
for num, img in enumerate(new_test):
    test_img = cv2.resize(cv2.imread(TEST_DIR + '/' + img), (50, 50))
    preprocessed_img = preprocess(TEST_DIR + '/' + img)
    img_data = preprocessed_img.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([img_data])[0]

    str_label = np.argmax(model_out)
    show = '{}'.format(str_label)

    y = fig.add_subplot(5, np.ceil(len(new_test)/5)+5, num+1)
    y.imshow(test_img, cmap = 'gray', interpolation = 'bicubic')
    plt.title(show)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()