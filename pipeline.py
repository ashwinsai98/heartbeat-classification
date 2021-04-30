import numpy as np
import librosa
import pickle



def pipeline():
    audio = "Sairam.wav"
    features = []
    x, sr = librosa.load(audio, 22050)
    x = crop(x, sr, 0.3)
    mfcc =  librosa.feature.mfcc(y=x, sr= sr, n_mfcc= 30)
    features.append([np.mean(x) for x in mfcc])
    features.append(sum(librosa.zero_crossings(x)))
    features.append(np.mean(librosa.feature.spectral_centroid(x)))
    features.append(np.mean(librosa.feature.spectral_rolloff(x,sr=sr)))
    features.append(np.mean(librosa.feature.chroma_stft(x,sr=sr)))
    
    mean = []
    var = []
    with open('standard_mean.txt', 'r') as file:
        for line in file:
            mean.append(int(line))
    with open('standard_var.txt', 'r') as file:
        for line in file:
            var.append(int(var))
    
    for i in range(0, len(features)):
        features[i] = (features[i] - mean[i])/var[i]

    model = pickle.load("Final_model.sav","rb")
    category = model.predict(features)
    if category == 0:
        return("artifact")
    elif category == 1:
        return("extrahls")
    elif category == 2:
        return("extrastole")
    elif category == 3:
        return("murmur")
    else:
        return("normal")



n_mfcc = 30
