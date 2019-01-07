import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import numpy as np

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))#output60*60*32
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))#60*60*64
model.add(MaxPooling2D(pool_size=(2, 2)))#30*30*64
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['acc'])

with open("tiger&elephant.pkl","rb") as f:
    x_train = pickle.load(f)
    y_train = pickle.load(f)
    x_test = pickle.load(f)
    y_test = pickle.load(f)
x_train = np.reshape(x_train,(800,120,120,1))
x_test = np.reshape(x_test,(100,120,120,1))
save_best = ModelCheckpoint("C:/Users/ASUS/Desktop/model_x.h5")#文件保存名以及路径
model.fit(x_train, y_train, batch_size=200, epochs=100,validation_data=(x_test,y_test),callbacks=save_best)#在正确率达到最高的时候保存
score = model.evaluate(x_test, y_test, batch_size=100)
print(score)