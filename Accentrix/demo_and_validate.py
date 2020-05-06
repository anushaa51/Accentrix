import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True # Disabling TensorFlow warnings

import os
import numpy as np

from keras.models import load_model

from preprocess import preprocess_single_file

IN_DIR = 'data/cmu_arctic/indian-english-male-ksp/wav/'
num_mfcc_coeffs = 25

# Start of execution

converter_model_name = input("Enter name of converter model to be used (with extension): ")
print("\n\n")

classifier_model_name = input("Enter name of classifier model to be used (with extension): ")
print("\n\n")

converter_model = None
classifier_model = None

try:
    print("Loading models...\n\n")
    converter_model = load_model("models/mfcc_converter/" + converter_model_name)
    classifier_model = load_model("models/mfcc_classifier/" + classifier_model_name)

except:
    print("Could not load models.")
    exit()

audio_name = input("\n\nEnter name of wav audio file to convert (in Accentrix directory, with extension): ")

inputs = None

try:
    inputs = list(preprocess_single_file(num_mfcc_coeffs, audio_name))
except:
    print("Could not open audio file.")
    exit()


inputs = np.reshape(inputs, [-1, 25])

predictions = []

print("\n\nBefore Conversion:\n")

for mfcc_vector in inputs:
    predictions += list(classifier_model.predict(x = np.array([mfcc_vector])))

total_US = 0
total_IN = 0

for prediction in predictions:
    total_US += prediction[0]
    total_IN += prediction[1]

print("Probability that MFCC vectors belong to US accent: ", (total_US/len(predictions))*100, " %")
print("Probability that MFCC vectors belong to Indian accent: ", (total_IN/len(predictions))*100, " %")


converted = []

print("\n\nConverting MFCC vectors of sample audio file from Indian to US accent...\n\n")

for mfcc_vector in inputs:
    converted += list(converter_model.predict(x = np.array([mfcc_vector])))

print("Done.")

converted = np.reshape(converted, [-1,25])

predictions = []

print("\n\nClassifying converted MFCC vectors...\n\n")

for mfcc_vector in converted:
    predictions += list(classifier_model.predict(x = np.array([mfcc_vector])))

print("Done\n\nAfter Conversion:\n")

total_US = 0
total_IN = 0

for prediction in predictions:
    total_US += prediction[0]
    total_IN += prediction[1]

print("Probability that MFCC vectors belong to US accent: ", (total_US/len(predictions))*100, " %")
print("Probability that MFCC vectors belong to Indian accent: ", (total_IN/len(predictions))*100, " %")

print("\n\n\n")


# Comment out rest of the script if you do not wish to validate on entire dataset 


print("Testing average conversion accuracy on entire dataset using the classifier")

inputs = []
for IN_fname in os.listdir(IN_DIR):
    mfcc_vectors = list(preprocess_single_file(num_mfcc_coeffs, IN_DIR + IN_fname))
    inputs += list(mfcc_vectors)

inputs = np.reshape(inputs, [-1, 25])

print("\n\nBefore Conversion:\n")

count = 0
total = len(inputs)
for mfcc_vector in inputs:
    count +=1
    print("Classifying ",count," of ",total," MFCC vectors", end="\r")
    predictions += list(classifier_model.predict(x = np.array([mfcc_vector])))

total_US = 0
total_IN = 0

for prediction in predictions:
    total_US += prediction[0]
    total_IN += prediction[1]

print("Average probability that MFCC vectors belong to US accent: ", (total_US/len(predictions))*100, " %")
print("Average probability that MFCC vectors belong to Indian accent: ", (total_IN/len(predictions))*100, " %")


converted = []

print("\n\nConverting MFCC vectors of all Indian audio clips in dataset...\n\n")

count = 0
total = len(inputs)
for mfcc_vector in inputs:
    count +=1
    print("Converting ",count," of ",total," MFCC vectors", end="\r")    
    converted += list(converter_model.predict(x = np.array([mfcc_vector])))

print("\n\nDone.")

converted = np.reshape(converted, [-1,25])

predictions = []

print("\n\nClassifying converted MFCC vectors...\n\n")

count = 0
total = len(converted)
for mfcc_vector in converted:
    count +=1
    print("Classifying ",count," of ",total," MFCC vectors", end="\r")
    predictions += list(classifier_model.predict(x = np.array([mfcc_vector])))

print("\n\nDone\n\nAfter Conversion:\n")

total_US = 0
total_IN = 0

for prediction in predictions:
    total_US += prediction[0]
    total_IN += prediction[1]

print("Average probability that MFCC vectors belong to US accent: ", (total_US/len(predictions))*100, " %")
print("Average probability that MFCC vectors belong to Indian accent: ", (total_IN/len(predictions))*100, " %")

# End of execution

