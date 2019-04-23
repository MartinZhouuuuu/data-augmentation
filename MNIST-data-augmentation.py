from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(ti,tl),(test_images,test_labels) = mnist.load_data()
test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Conv2D(32,(5,5), activation = 'relu', 
	input_shape = (28,28,1)))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(5,5),activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
	rotation_range = 20,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	rescale =  1./255,
	shear_range = 0.2,
	zoom_range = 0.2,
	fill_mode = 'nearest'
	)
train_generator = train_datagen.flow_from_directory(
	'/Users/apple/Desktop/trainingSet',
	target_size = (28,28),
	batch_size = 16,
	class_mode = 'categorical',
	color_mode = 'grayscale')
'''test_datagen = ImageDataGenerator(
	rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
	'/Users/apple/Desktop/testSet',
	target_size = (28.28),
	batch_size = 16,
	class_mode = 'categorical',
	color_mode = 'grayscale')'''
model.fit_generator(
	train_generator,
	steps_per_epoch = 3000,
	epochs = 10
	)

test_loss, test_accuracy = model.evaluate(test_images,test_labels)