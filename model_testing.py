import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncode,StandardScaler
import pandas as pd
from sklearn.svm import SVC
import joblib
import sounddevice as sd
import numpy as np
import librosa
import time
import keyboard
import pygame
import time

pygame.init()
duration = 3 
sample_rate = 22050  

while True:
    print("litening...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait() 
    audio = audio.flatten()
    #sd.play(audio, samplerate=sample_rate)  
    #print("Recording finished")

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=30)
    scaled_mfccs = np.mean(mfccs.T, axis=0).reshape(1,-1)  

    # here enter your directory path
    model=joblib.load('') 
   
    scaler=joblib.load('')
    
    scaled_featurs=scaler.transform(scaled_mfccs)
    pred = model.predict(scaled_featurs)
    print("Predicted class:", result)
    
    if result[0]=='Aman':
        '''keyboard.press('ctrl+d')
        time.sleep(15)
        keyboard.press('ctrl+d')'''

        print("Aman")

        '''pygame.mixer.music.load('D:\\spotyfy_python\\ji.mp3')
        pygame.mixer.music.play()'''
        


        





