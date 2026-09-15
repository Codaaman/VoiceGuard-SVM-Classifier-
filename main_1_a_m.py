import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pandas as pd
from sklearn.svm import SVC
import joblib
feature=[] 
lable=[]



path='D:\\spotyfy_python\\audio_aynaliss\\dataset'
labels=['Aman','Music']


for lab in labels:
    fol=os.path.join(path,lab)
    for file in os.listdir(fol):
            full_path=os.path.join(fol,file)
            #feaatur_extractor(full_path,lab)
            data,sample_rate=librosa.load(full_path)
            mfcc=librosa.feature.mfcc(y=data,sr=sample_rate,n_mfcc=50)
            scale_mfcc=np.mean(mfcc.T,axis=0)
            lable.append(lab)
            feature.append(scale_mfcc)


    

df=pd.DataFrame({"Audio_featurs":feature,"Labels":lable})
x=np.array(df['Audio_featurs'].to_list())
y=LabelEncoder().fit_transform(df["Labels"])

scaler=StandardScaler()
x=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42,stratify=y)




#svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model = SVC(kernel='rbf', C=10, gamma='scale')
svm_model.fit(x_train,y_train)  # linear kernel
y_pred=svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler,"scaler_modle.plk")







