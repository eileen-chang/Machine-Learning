from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


INPUT_SIZE = 3 * 32 * 3
NUM_CLASSES = 10
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


def deep_learning_with_keras():
    # Convert class vectors to binary class matrices.
    (x_trian, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES)


    x_train, x_val, y_train, y_val = train_test_split(x_trian, y_train, test_size = 0.2, random_state = 1)

    plt.imshow(x_train[0])
    plt.figure(figsize=(0.5,0.5))

    # We will also convert the images to float and will divided by 255 so that the values are between (0,1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')

    x_train /= 255
    x_test /= 255
    x_val /= 255

    model_A = netA()
    (A_model, A_history) = train_model(model_A, x_train, y_train, x_val, y_val)

    model_B = netB()
    (B_model, B_history) = train_model(model_B, x_train, y_train, x_val, y_val)

    model_C = netC()
    (C_model, C_history) = train_model(model_C, x_train, y_train, x_val, y_val)

    model_D = netD()
    (D_model, D_history) = train_model(model_D, x_train, y_train, x_val, y_val)

    result_A = model_A.evaluate(x_test, y_test, batch_size=32)
    result_B = model_B.evaluate(x_test, y_test, batch_size=32)
    result_C = model_C.evaluate(x_test, y_test, batch_size=32)
    result_D = model_D.evaluate(x_test, y_test, batch_size=32)

    plt.figure(figsize=(16,10))
    epoch = []
    for i in range(1,11):
        epoch.append(i)
    
    A_train = plt.plot(epoch, A_history.history['accuracy'], 'r', label='A_train')
    A_valication = plt.plot(epoch, A_history.history['val_accuracy'], 'r--', label='A_validaiton')

    B_train = plt.plot(epoch, B_history.history['accuracy'], 'b', label='B_train')
    B_valication = plt.plot(epoch, B_history.history['val_accuracy'], 'b--', label='B_validaiton')

    C_train = plt.plot(epoch, C_history.history['accuracy'], 'g', label='C_train')
    C_valication = plt.plot(epoch, C_history.history['val_accuracy'], 'g--', label='C_validaiton')

    D_train = plt.plot(epoch, D_history.history['accuracy'], 'purple', label='D_train')
    D_valication = plt.plot(epoch, D_history.history['val_accuracy'], color='purple', linestyle='--', label='D_validaiton')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.xlim([1,NUM_EPOCHS])
    plt.title('Model Accuracy')
    fname = 'Model_Accuracy'
    plt.savefig(fname)





def train_model(model, x_train, y_train, x_val, y_val, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    # We will train the model with the RMSprop algorithm
    opt = keras.optimizers.RMSprop(learning_rate=learning_rate, decay=1e-6)

    #We will use the crossentropy cost, and will look at the accuracy
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    #Here we fit the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val), shuffle=False)

    return (model,history)


def netA():
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(10, activation='softmax'))
    return (model)

def netB():
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return (model)

def netC():
    model = Sequential()
    model.add(Conv2D(25, (5,5), activation='relu', padding='same', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    return (model)

def netD():
    model = Sequential()
    model.add(Conv2D(25, (5,5), activation='relu', padding='same', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2,2), strides=2))
    model.add(Conv2D(25, (5,5), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return (model)


if __name__=='__main__':
    deep_learning_with_keras()
