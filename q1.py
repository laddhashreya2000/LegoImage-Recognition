from pandas import read_csv
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
def encode(arr):
	# create empty vector
	encoding = []
	# mark 1 for each tag in the vector
	for i in range(len(arr)):
		encoding.append(str(chr(97+int(arr[i]))))
	return encoding
data = read_csv("./Train.csv")
train_df, valid_df = train_test_split(data, test_size=0.2, random_state=2)
valid = valid_df
train_df.category = encode(np.array(train_df.category))
valid_df.category = encode(np.array(valid_df.category))
train_datagen = ImageDataGenerator(rescale=1./255)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='./train',
        x_col="name",
        y_col="category",
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

validation_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='./train',
        x_col="name",
        y_col="category",
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

cnn4 = Sequential()
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
#cnn4.add(BatchNormalization())

#cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.25))

#cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#cnn4.add(BatchNormalization())
#cnn4.add(MaxPooling2D(pool_size=(2, 2)))
#cnn4.add(Dropout(0.25))

cnn4.add(Flatten())

#cnn4.add(Dense(512, activation='relu'))
#cnn4.add(BatchNormalization())
#cnn4.add(Dropout(0.5))

cnn4.add(Dense(128, activation='relu'))
#cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(16, activation='softmax'))

cnn4.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

cnn4.fit_generator(
        train_generator,
        steps_per_epoch=len(train_df),
        epochs=2, validation_data=validation_generator)

from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from csv import writer


def validate(arr):
    Y_pred =[]
    for filename in arr:
        photo = load_img('train/'+filename, target_size=(64,64))
        photo = img_to_array(photo, dtype='uint8')
        photo = np.expand_dims(photo, axis=0)
        Y=cnn4.predict(photo)
        o = np.where(Y == np.amax(Y))
        Y_pred.append(o[1][0]+1)
    return Y_pred

Y_valid = validate(valid.name)
Y_actual = valid.category.to_list()
Score = 100*f1_score(Y_actual, Y_valid, average='weighted')


cnn4.save('./models/model.h5')
cnn4.save_weights('./models/weights.h5')

from os import listdir

def load_dataset(path):
    Y_pred = []
    name =[]
    for filename in listdir(path):
        photo = load_img(path+filename, target_size = (64,64))
        photo = img_to_array(photo, dtype='uint8')
        photo = np.expand_dims(photo, axis=0)
        Y = cnn4.predict(photo)
        o = np.where(Y == np.amax(Y))
        Y_pred.append(o[1][0]+1)
        name.append(filename)
    return Y_pred,name

Y_pred, name= load_dataset('test/')

pd.DataFrame({'name': name, 'category': Y_pred}).to_csv('Submission.csv', index=False)