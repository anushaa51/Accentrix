After doing a lot of variations of the network, changing all sorts of activation functions, optimizers, hyperparameters, etc,
I settled on 3 types, all types have xavier initialised network weights (also known as glorot):

relu as activation function in hidden layers, with batch normalisation after non-linear activation functions, optimised using SGD
tanh as activation function in hidden layers, 0 initialised bias for all layers, optimised using Adam
tanh as activation function in hidden layers, without bias in any layer, optimised using Adam
tanh as activation function in hidden layers, 0 initialised bias for all layers, normalised inputs before training using StandardScaler from scikit, optimised using Adam


I proceeded to try to overfit the models to see which model had best learning capacity.
the 4th one had the best accuracy after all 4 were trained for 2000 epochs.

I then proceed to run the 4th model for different number of epochs, doing a sort of binary search. 
I found that 1500 epochs had the best accuracy of 95.98%

The graphs for this model have the term "FINAL" in the name in the graphs folder, along with other relevant graphs.



Note: In case you're getting a key error when trying to train a model, in all the different model codes, you might have to find and replace 
history.history['acc']  => history.history['accuracy']
and
history.history['val_acc']  => history.history['val_accuracy']