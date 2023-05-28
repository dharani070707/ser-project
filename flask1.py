import pickle
import soundfile
import numpy as np
import librosa
from flask import Flask, request, render_template
import pickle

global x
app = Flask(__name__, template_folder='template',static_folder='static')
path = '.\\uploads'
filename = 'C:\\Users\\DharaniPrasadS\\Desktop\\dharani\\models\\mymodel.pkl'
loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method == 'POST':
        file = request.files['file']
        file.save('C:\\Users\\DharaniPrasadS\\Desktop\\dharani\\uploads\\file.wav')
        return render_template('index.html')
    
@app.route('/predict',methods=['GET','POST'])
def predict():
    print('checking')
    loaded_model = pickle.load(open('C:\\Users\\DharaniPrasadS\\Desktop\\dharani\\models\\mymodel.pkl', 'rb')) # loading the model file from the storage
    feature=extract_feature("C:\\Users\\DharaniPrasadS\\Desktop\\dharani\\uploads\\file.wav", mfcc=True, chroma=True, mel=True)
    feature=feature.reshape(1,-1)
    predict1 = loaded_model.predict(feature)
    global x
    x = predict1
    path = '/static/css/images/avatars/'+predict1[0]+'.jpg'
    return render_template('index.html', prediction_text = 'The voice mostly had '+predict1[0]+' emotion',path1 = path )

if __name__=="__main__":
    app.run(debug=True)

    


    


