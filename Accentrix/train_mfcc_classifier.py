import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True # Disabling TensorFlow warnings


import os
import random

import numpy as np

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from preprocess import preprocess_single_file


US_DIR = 'data/cmu_arctic/us-english-male-bdl/wav/'
IN_DIR = 'data/cmu_arctic/indian-english-male-ksp/wav/'

RANDOM_SEED = 42


# Start of execution

batch_size = 50000  # Number of MFCC vectors to propogate before updating weights (mini batching)
n_epochs = int(input("Enter number of epochs: "))
lr = 1e-3
num_mfcc_coeffs = 25

state_size_1 = 25
state_size_2 = 50
state_size_3 = 50
state_size_4 = 25

US_inputs = []
US_labels = []
IN_inputs = []
IN_labels = []

# Preprocessing and adding labels to each file, high value of first index in the label indicates a US accent, and high value of second index indicates Indian accent.

print("\n\nPreprocessing and constructing the dataset, please wait...")

for US_fname, IN_fname in zip(os.listdir(US_DIR), os.listdir(IN_DIR)):

    mfcc_vectors = preprocess_single_file(num_mfcc_coeffs, US_DIR + US_fname)

    US_inputs += list(mfcc_vectors)

    US_labels += [[1.0, 0.0]]*len(mfcc_vectors)


    mfcc_vectors = preprocess_single_file(num_mfcc_coeffs, IN_DIR + IN_fname)

    IN_inputs += list(mfcc_vectors)

    IN_labels += [[0.0, 1.0]]*len(mfcc_vectors)


inputs = US_inputs + IN_inputs
labels = US_labels + IN_labels

# Merging and randomising indices

randomized_indices = list(range(0, len(inputs))) 
random.shuffle(randomized_indices)

inputs = [inputs[i] for i in randomized_indices]
labels = [labels[i] for i in randomized_indices]

# Construction of classifier

model = Sequential()

model.add(Dense(state_size_1, input_shape = (25,))) 
model.add(Activation('relu'))

model.add(Dense(state_size_2))
model.add(Activation('relu'))

model.add(Dense(state_size_3))
model.add(Activation('relu'))

model.add(Dense(state_size_4))
model.add(Activation('relu'))
          
model.add(Dense(2))
model.add(Activation('softmax'))

optimizer = Adam(lr = lr)
model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

print("\n\n\n\n\nSummary of Model: \n\n")
model.summary()


inputs = np.reshape(inputs, [-1, 25])
labels = np.reshape(labels, [-1, 2])


X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=RANDOM_SEED)


print("\n\n\n\n\nStarting to train model...\n\n\n")
history = model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = n_epochs, verbose = 1, validation_data = (X_test,y_test))
print("\n\n\nDone train model.\n\n\n")

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.title('Classifier Model Acc')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.show()   # Plot of improvement in accuracy of model vs epoch


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show() # Plots of loss and accuracy for both training and validation data vs epoch

predictions = model.predict(x = X_test)
correct = 0

for i in range(len(predictions)):
    if (predictions[i][0] > predictions[i][1] and y_test[i][0] > y_test[i][1]) or (predictions[i][1] > predictions[i][0] and y_test[i][1] > y_test[i][0]):
        correct += 1

print("\n\nCorrectly classified ",str(correct)," MFCC vectors out of ",str(len(predictions)), " MFCC vectors. ( ", str((correct/len(predictions))*100), " % )" )

#name = str(input("\n\nSaving Model, Enter a name to save as: "))

#model.save("models/mfcc_classifier/" + name + ".h5")
import os

name = ""
cond = True
while cond:
    name = str(input("\n\nSaving Model, Enter a name to save as: "))
    for file_name in os.listdir("models/mfcc_classifier/"):
        if name == file_name:
            print("File already exists, pick another name")
            continue
    cond = False
        
try:
        model.save("models/mfcc_classifier/" + name + ".h5")
except:
    print("File system is busy, could not save model")


del model

# End of execution
