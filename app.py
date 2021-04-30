import pickle
import librosa
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "AudioFolder"

### Function to shift the data along time axis
def manipulate(data, sr, time, direction):
    shift = int(sr*time)
    if direction == 'right':
        shift = -shift
    aug_data = np.roll(data,shift)
    if shift > 0:
        aug_data[:shift] = 0
    else:
        aug_data[shift:] = 0
    return aug_data


### Function to chop initial and end parts of the audio file
def crop(data, sr, time):
    data = manipulate(data, sr, time, 'right')
    data = manipulate(data, sr, time*2, 'left')
    data = manipulate(data, sr, time, 'right')
    return data

def pipeline():
    audio = "Sairam.wav"
    features = []
    x, sr = librosa.load(audio, 22050)
    x = crop(x, sr, 0.3)
    mfcc =  librosa.feature.mfcc(y=x, sr= sr, n_mfcc= 30)
    mfccs = [np.mean(x) for x in mfcc]
    for i in mfccs:
        features.append(i)
    features.append(sum(librosa.zero_crossings(x)))
    features.append(np.mean(librosa.feature.spectral_centroid(x)))
    features.append(np.mean(librosa.feature.spectral_rolloff(x,sr=sr)))
    features.append(np.mean(librosa.feature.chroma_stft(x,sr=sr)))
    '''
    mean = []
    var = []
    with open('standard_mean.txt', 'r') as file:
        for line in file:
            line = line[0:-1]
            mean.append(float(line))
    with open('standard_var.txt', 'r') as file:
        for line in file:
            line = line[0:-1]
            var.append(float(line))
    
    print(features)

    #for i in range(0, len(features)):
     #   features[i] = (features[i] - mean[i])/np.sqrt(var[i])
    #print(features)
'''
    features = np.array(features)
    model = pickle.load(open("Final_model.sav", "rb"))
    category = model.predict(features.reshape(1,-1))
    if category == 0:
        return("Artifact")
    elif category == 1:
        return("Extrahls")
    elif category == 2:
        return("Extrastole")
    elif category == 3:
        return("Murmur")
    elif category == 4:
        return("Normal")



@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route('/index', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      
      f = request.files['file']
      f.save("Sairam.wav")
      category = pipeline()
      return render_template('index.html', resu = category)


if __name__ == "__main__":
    app.run(debug=True)
