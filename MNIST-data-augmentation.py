from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from imgaug import augmenters as iaa
(ti,tl),(test_images,test_labels) = mnist.load_data()
test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

model = Sequential()
#convolution
model.add(Conv2D(32,(5,5), activation = 'relu', 
	input_shape = (28,28,1)))
#model.add(BatchNormalization())
model.add(Conv2D(32,(5,5), activation = 'relu', ))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(5,5),activation = 'relu'))
#model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
#convert 3D tensor to 1D
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

aug_1 = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
aug_2 = iaa.Invert(1)
def augmentation(image):
	image = aug_1.augment_image(image)
	image = aug_2.augment_image(image)
	return image

train_datagen = ImageDataGenerator(
	rotation_range = 0.2,
	rescale =  1./255,
	shear_range = 0.05,
	zoom_range = 0.05,
	fill_mode = 'nearest',
	featurewise_center = True,
	featurewise_std_normalization = True
	)
train_generator = train_datagen.flow_from_directory(
	'trainingSet',
	target_size = (28,28),
	batch_size = 1,
	class_mode = 'categorical',
	color_mode = 'grayscale',
	)
test_datagen = ImageDataGenerator(
	rescale = 1./255,
	preprocessing_function = augmentation)

test_generator = test_datagen.flow_from_directory(
	'testSet',
	target_size = (28,28),
	batch_size = 1,
	class_mode = 'categorical',
	color_mode = 'grayscale',
	save_to_dir = 'test_aug')

model.fit_generator(
	train_generator,
	steps_per_epoch = 60000,
	epochs = 30,
	validation_data = test_generator,
	validation_steps = 100
	)

