import numpy as np
import os
import random

import scipy.io.wavfile as wav
from python_speech_features import mfcc

from fast_dtw import get_dtw_series


def preprocess_data(num_mfcc_coeffs, SOURCE_DIR, TARGET_DIR):

    """
    Aligning source and target frames before training
    """
    
    inputs = [] 
    labels = [] 

    counter = 1
    total = len(os.listdir(TARGET_DIR))
    
    for source_fname, target_fname in zip(os.listdir(SOURCE_DIR), os.listdir(TARGET_DIR)):

        (source_sample_rate, source_wav_data) = wav.read(SOURCE_DIR + source_fname) 
        (target_sample_rate, target_wav_data) = wav.read(TARGET_DIR + target_fname)

        source_mfcc_features = np.array(mfcc(source_wav_data, samplerate=source_sample_rate, numcep=num_mfcc_coeffs))
        target_mfcc_features = np.array(mfcc(target_wav_data, samplerate=target_sample_rate, numcep=num_mfcc_coeffs))

        # Aligns the MFCC features matrices using FastDTW.
        source_mfcc_features, target_mfcc_features = get_dtw_series(source_mfcc_features, target_mfcc_features, counter, total)
        counter += 1

        inputs += list(source_mfcc_features) 

        labels += list(target_mfcc_features) 

    randomized_indices = list(range(0, len(inputs))) 
    random.shuffle(randomized_indices)

    inputs = [inputs[i] for i in randomized_indices]

    labels = [labels[i] for i in randomized_indices]

    return inputs, labels



def preprocess_single_file(num_mfcc_coeffs, audio_file):

    """
    FastDTW is not required, as we're not trying to align source and target frames, we're just predicting
    """

    source_sample_rate, source_wav_data = None, None
    
    try:
        (source_sample_rate, source_wav_data) = wav.read(audio_file)
    except:
        raise("Can't open file")

    source_mfcc_features = np.array(mfcc(source_wav_data, samplerate=source_sample_rate, numcep=num_mfcc_coeffs))

    return source_mfcc_features