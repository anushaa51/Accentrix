import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True # Disabling TensorFlow warnings

import numpy as np

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.initializers import glorot_uniform, constant
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocess import preprocess_data


TARGET_DIR = 'data/cmu_arctic/us-english-male-bdl/wav/'
SOURCE_DIR = 'data/cmu_arctic/indian-english-male-ksp/wav/'

RANDOM_SEED = 42


# Start of execution

batch_size = 512 # 30000  # Number of MFCC vectors to propogate before updating weights (mini batching)
n_epochs = int(input("Enter number of epochs: "))
lr = 1e-4
num_mfcc_coeffs = 25

state_size_1 = 25
state_size_2 = 100
state_size_3 = 100
state_size_4 = 25


import pickle

inputs, labels = None, None

try:
    with open("inputs.dat", "rb") as f:
        inputs = pickle.load(f)
    with open("labels.dat", "rb") as f:
        labels = pickle.load(f)
except:
    print("\nNo cache found, preprocessing and caching\n")

if inputs == None:
    inputs, labels = preprocess_data(num_mfcc_coeffs, SOURCE_DIR, TARGET_DIR)
    with open("inputs.dat", "wb") as f:
        pickle.dump(inputs, f, pickle.HIGHEST_PROTOCOL)
    with open("labels.dat", "wb") as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)



model = Sequential()

initializer = glorot_uniform(seed=RANDOM_SEED) # Xavier uniform weights initialisation
bias = constant(value=0)

model.add(Dense(state_size_1, input_shape = (25,), kernel_initializer=initializer, bias_initializer=bias))

model.add(Dense(state_size_2, kernel_initializer=initializer, bias_initializer=bias))
model.add(Activation("tanh"))

model.add(Dense(state_size_3, kernel_initializer=initializer, bias_initializer=bias))
model.add(Activation("tanh"))

model.add(Dense(state_size_4, kernel_initializer=initializer, bias_initializer=bias))

optimizer = Adam(lr = lr)
model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

print("\n\n\n\n\nSummary of Model: \n\n")
model.summary()


inputs = np.reshape(inputs, [-1, 25])
labels = np.reshape(labels, [-1, 25])


X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=RANDOM_SEED)

scaler = StandardScaler()
scaler.fit(inputs)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


print("\n\n\n\n\nStarting to train model...\n\n\n")
history = model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = n_epochs, verbose = 1, validation_data = (X_test,y_test))
print("\n\n\nDone train model.\n\n\n")

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.title('tanh Adam 0-bias normalised input Converter Model Acc')
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
diff = []

for i in range(len(predictions)):
    diff.append(sum(abs(predictions[i]-y_test[i])))
    print("L1 Norm b/w predicted MFCC vector and validation MFCC vector " + str(i) + ": ", diff[i], end="\r")

print("\n\nCumulative L1 Norm b/w predicted and validation MFCC vectors (",str(len(diff))," MFCC vectors): ",sum(diff))

name = str(input("\n\nSaving Model, Enter a name to save as: "))

model.save("models/mfcc_converter/" + name + ".h5")

del model


# End of execution
