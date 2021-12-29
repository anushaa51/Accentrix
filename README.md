# Accentrix
Web based Machine Learning application which performs accent conversion and classification through mapping of MFCCs.

# Synopsis
With speech recognition slowly becoming ubiquitous and all pervasive technology, there
arises a need to eliminate the challenges faced by speech recognition systems. The large
number of existing languages makes speech recognition a technology that requires extensive amounts of data for building the model or software. To add to this conundrum, most
languages have a sizable number of accents being spoken depending on the regions. This
variation in accents and the variations of speech in itself are a formidable challenge standing
in the way of accurate speech recognition.
To tackle this issue, there are two possible solutions in sight. The first being, training speech
recognition models for each accent in each language. This solution invariably requires large
amounts of data for each accent, as well as extensive investments in resources such as time
and computing power. This approach is viable for large companies which are able to source
such resources, however the data requirement might still remain as an issue.
The second solution is to train speech recognition models that simply convert the features
of one accent to that of another target accent, wherein there already exists a fully trained
model for speech recognition of the target accent. This solution reduces both, the amount
of data required to train models, as well the amount of training required since only one fully
trained speech recognition model is required per language.
The proposed solution is thus a system which uses features of a voice called MFCCs, i.e.
Mel Frequency Cepstral Coefficients, which are a vector of features that together describe
the short-term power spectrum of a sound. MFCCs of the source accent are extracted
from an audio file and a neural network is used to convert these MFCCs into that of the
target accent. In order to obtain the accuracy of this transformation, a classifier is built
which provides the result pertaining to how much the converted MFCCs resemble the target
accent.

# DataFlow Level 1

<img src="/assets/4.JPG?raw=true"/>

# Model Performance

<img src="/assets/5.JPG?raw=true"/>
<img src="/assets/2.JPG?raw=true"/>

# Screenshots

<img src="/assets/1.jpg?raw=true"/>
<img src="/assets/3.jpg?raw=true"/>
