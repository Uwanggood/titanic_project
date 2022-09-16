import os
import random
import platform
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, \
	BatchNormalization
from sklearn.model_selection import train_test_split


is_window = platform.system() == 'Windows'
splitter = '\\' if is_window else '/'

if is_window:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def run(
	data_dir=os.path.dirname(os.getcwd()) + splitter,
	fast_run=False):
	TRAIN_PATH = data_dir + f'{splitter}train{splitter}'
	TEST_PATH = data_dir + f'{splitter}test1{splitter}'
	print(os.listdir(data_dir))

	# Define Constatnts
	IMAGE_WIDTH = 128
	IMAGE_HEIGHT = 128
	IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
	IMAGE_CHANNELS = 3

	filenames = os.listdir(TRAIN_PATH)
	categories = []
	for filename in filenames:
		category = filename.split('.')[0]
		if category == 'dog':
			categories.append(1)
		else:
			categories.append(0)

	df = pd.DataFrame({
		'filename': filenames,
		'category': categories
	})

	df['category'].value_counts().plot.bar()
	plt.show()

	sample = random.choice(filenames)
	image = load_img(TRAIN_PATH + sample)
	plt.imshow(image)

	model = Sequential()

	# param_number = output_channel_number * (input_channel_number * kernel_height * kernel_width + 1)
	# 32 * (3 * 3 * 3 + 1)
	model.add(
		Conv2D(32, (3, 3), activation='relu',
			   input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# 64 * ( 32 * 3 * 3 + 1
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))  # 2 because we have cat and dog classes

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	model.summary()

	earlystop = EarlyStopping(patience=10)

	learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
												patience=2,
												verbose=1,
												factor=0.5,
												min_lr=0.00001)

	callbacks = [earlystop, learning_rate_reduction]

	df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

	train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
	train_df = train_df.reset_index(drop=True)
	validate_df = validate_df.reset_index(drop=True)

	train_df['category'].value_counts().plot.bar()
	plt.show()
	validate_df['category'].value_counts().plot.bar()
	plt.show()

	total_train = train_df.shape[0]
	total_validate = validate_df.shape[0]
	batch_size = 15

	train_datagen = ImageDataGenerator(
		rotation_range=15,
		rescale=1. / 255,
		shear_range=0.1,
		zoom_range=0.2,
		horizontal_flip=True,
		width_shift_range=0.1,
		height_shift_range=0.1
	)

	train_generator = train_datagen.flow_from_dataframe(
		train_df,
		TRAIN_PATH,
		x_col='filename',
		y_col='category',
		target_size=IMAGE_SIZE,
		class_mode='categorical',
		batch_size=batch_size
	)

	validation_datagen = ImageDataGenerator(rescale=1. / 255)
	validation_generator = validation_datagen.flow_from_dataframe(
		validate_df,
		TRAIN_PATH,
		x_col='filename',
		y_col='category',
		taraget_size=IMAGE_SIZE,
		class_mode='categorical',
		batch_size=batch_size
	)

	example_df = train_df.sample(n=1).reset_index(drop=True)
	example_df = train_datagen.flow_from_dataframe(
		example_df,
		TRAIN_PATH,
		x_col='filename',
		y_col='category',
		target_size=IMAGE_SIZE,
		class_mode='categorical'
	)

	plt.figure(figsize=(12, 12))
	for i in range(0, 15):
		plt.subplot(5, 3, i + 1)
		for X_batch, Y_batch in example_df:
			image = X_batch[0]
			plt.imshow(image)
			break

	epochs = 3 if fast_run else 50
	history = model.fit_generator(
		train_generator,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=total_validate // batch_size,
		steps_per_epoch=total_train // batch_size,
		callbacks=callbacks
	)

	model.save_weights("model.h5")

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
	ax1.plot(history.history['loss'], color='b', label="Training loss")
	ax1.plot(history.history['val_loss'], color='r', label="validation loss", axes=ax1)
	ax1.set_xticks(np.arange(1, epochs, 1))
	ax1.set_yticks(np.arange(0, 1, 0.1))

	ax2.plot(history.history['acc'], color='b', label="Training accuracy")
	ax2.plot(history.history['val_acc'], color='r', label="Validation accuracy")
	ax2.set_xticks(np.arange(1, epochs, 1))

	legend = plt.legend(loc='best', shadow=True)
	plt.tight_layout()
	plt.show()

	# Prepare Testing Data
	test_filenames = os.listdir(TEST_PATH)
	test_df = pd.DataFrame({
		'filename': test_filenames
	})
	nb_samples = test_df.shape[0]

	# Create Testing Generator
	test_gen = ImageDataGenerator(rescale=1. / 255)
	test_generator = test_gen.flow_from_dataframe(
		test_df,
		TEST_PATH,
		x_col='filename',
		y_col=None,
		class_mode=None,
		target_size=IMAGE_SIZE,
		batch_size=batch_size,
		shuffle=False
	)

	# Predict
	predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples / batch_size))

	test_df['category'] = np.argmax(predict, axis=-1)

	label_map = dict((v, k) for k, v in train_generator.class_indices.items())
	test_df['category'] = test_df['category'].replace(label_map)

	test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})

	test_df['category'].value_counts().plot.bar()

	sample_test = test_df.head(18)
	sample_test.head()
	plt.figure(figsize=(12, 24))
	for index, row in sample_test.iterrows():
		filename = row['filename']
		category = row['category']
		img = load_img(TEST_PATH + filename, target_size=IMAGE_SIZE)
		plt.subplot(6, 3, index + 1)
		plt.imshow(img)
		plt.xlabel(filename + '(' + "{}".format(category) + ')')
	plt.tight_layout()
	plt.show()

	test_df.to_csv("submission.csv", index=False)


def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default=os.path.dirname(os.getcwd()) + splitter,
						help='data directory')
	parser.add_argument('--fast_run', type=bool, default=False,
						help='choose to run the code in fast mode or not')
	opt = parser.parse_args()
	return opt


def main(opt):
	run(**vars(opt))


if __name__ == "__main__":
	opt = parse_opt()
	main(opt)
