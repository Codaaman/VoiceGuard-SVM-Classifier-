# VoiceGuard-SVM-Classifier 

A high-efficiency, robust **Speaker Identification and Audio Classification pipeline** built using Python, Librosa, and Scikit-Learn. This framework extracts advanced spectral audio features to accurately perform binary classification between a specific speaker's voice (**"Aman"**) and ambient background **"Music"**.

---

Architectural Architecture & Flow

* Advanced Feature Representation: Utilizes `librosa` to extract 50 Mel-Frequency Cepstral Coefficients (MFCCs) from raw temporal audio waves, capturing deep vocal tract characteristics. 
* Statistical Feature Aggregation: Compresses multi-frame time-series matrices into a single static feature vector per audio file using time-axis mean feature scaling (`np.mean(mfcc.T, axis=0)`).
* Supervised Learning Engine: Classifies vector inputs using a fine-tuned Support Vector Machine (SVM) configured with a non-linear Radial Basis Function (RBF Kernel**, `C=10`), optimized for high boundary margin definition.
* Production Serialization: Pipelines include standard data normalization templates (`StandardScaler`), serialized via `joblib` for zero-lag real-time inference serving.

---

# Codebase Blueprint

* `audio_classifier.py`: Core pipeline script that automates dataset parsing, transforms wave signals to MFCC matrices, runs data splits, and trains the SVM model.
* `svm_model.pkl`: Trained Support Vector Machine classification model checkpoint weights.
* `scaler_model.pkl`: Serialized configuration data for data standardizers to ensure production feature parity.

---

# Tech Stack

* Audio Signal Processing: Librosa
* Statistical Learning Engine: Scikit-Learn (SVM, StandardScaler, LabelEncoder)
* Data Processing Layer: Pandas, NumPy
* Model Preservation Deployment: Joblib
