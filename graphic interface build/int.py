# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'int.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon

from scipy import stats

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Nadam

import librosa
import numpy as np
import pandas as pd

import sklearn as skl
import warnings
import pickle


genre_dict = {
    'Hip-Hop': 0,
    'Pop': 1,
    'Folk': 2,
    'Rock': 3,
    'Experimental': 4,
    'International': 5,
    'Electronic': 6,
    'Instrumental': 7,
}

inv_genre_dict = {v: k for k, v in genre_dict.items()}

genre_names = [inv_genre_dict[i] for i in range(8)]
genre_names

def prediction_confidence(X, model):
    c = max(model.predict(X)[0])
    if c < 0.25:
        return "Pas sur mais je pense que ca soit du "
    elif c < 35:
        return "On sent le "
    elif c < 45:
        return "Je suis un peu confiant que ca soit "
    elif c < 60:
        return "Je suis confiant que ca soit "
    elif c < 70:
        return "Je suis tres sur que ca soit "
    return "100% sur que c'est du "

def predict_class(X, model):
    return inv_genre_dict[model.predict_classes(X)[0]]

def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()

def compute_features(filepath):
    features = pd.Series(index=columns(), dtype=np.float32, name=filepath)

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        x, sr = librosa.load(filepath, sr=None, mono=True,duration=30.0)  # kaiser_fast

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rmse(S=stft)
        feature_stats('rmse', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print('{}: {}'.format(filepath, repr(e)))

    return features

def update_prediction(s, filler_track, pca, model):
    X = extract_features(s, filler_track, pca)
    p = model.predict_classes(X)
    c = prediction_confidence(X, model)
    print(c,'',genre_names[p[0]])
    

def extract_features(wavfile, filler_track, pca):
    X = compute_features(wavfile).to_frame().T
    X2 = filler_track.to_frame().T

    X = pd.concat([X,X2])
    X = skl.preprocessing.StandardScaler().fit_transform(X)
    X = pca.transform(X)
    print('Nombre de caracteristiques extraites est : '+ str(X.shape[1]))
    return X

model = Sequential()
model.add(Dense(512, input_dim=215, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))

model.add(Dense(8, activation='softmax'))

model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.load_weights('DNN3.h5')
    
pca = pickle.load(open('pca','rb'))


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("\n"
"background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(330, 340, 131, 41))
        self.pushButton.setStyleSheet("border-style:groove;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"color:white;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 280, 91, 31))
        self.pushButton_2.setStyleSheet("border-style:groove;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"color:white;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(260, 80, 281, 161))
        self.textBrowser.setStyleSheet("border-style:groove;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"background-color: rgb(255, 255, 255);")
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def pushButton_handler(self):
        print("button pressed")
        self.open_dialog_box()

    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        filler_track = compute_features(filename[0])

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Submit"))
        self.pushButton_2.setText(_translate("MainWindow", "Choose file"))
        self.pushButton_2.clicked.connect(self.pushButton_handler)
        
        self.pushButton.clicked.connect(update_prediction(filename[0],filler_track,pca,model))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
