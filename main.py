from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import IPython.display as ipd
import librosa, librosa.display
import math
import numpy as np
from collections import OrderedDict
import os
from PIL import Image, ImageDraw, ImageOps
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import statistics
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import warnings
import wave
from scipy.fftpack import dct
import scipy.io.wavfile as wav
from python_speech_features.sigproc import preemphasis, framesig
import argparse
import cv2

app = Flask(__name__)


def get_in_features(model):
    """Get the number of in_features of the classifier
    """
    in_features = 0
    for module in model.classifier.modules():
        try:
            in_features = module.in_features
            break
        except AttributeError:
            pass
    return in_features


def create_classifier(model, out_features, hidden_units=512):
    """Create the classifier
    """
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(get_in_features(model), hidden_units)),
        ('relu1', nn.ReLU(inplace=True)),
        ('drop1', nn.Dropout(.5)),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu2', nn.ReLU(inplace=True)),
        ('drop2', nn.Dropout(.5)),
        ('fc3', nn.Linear(hidden_units, out_features)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

NFFT = 512
PREEMPH = 0.97
HAMMING_WINFUNC = np.hamming
LIFTER = 6
ALPHA = 0.4
GAMMA = 0.9
sr=48000

def get_complex_spec(wav_, winstep, winlen, with_time_scaled=False):
    """Return complex spec
    """
    sig,rate = librosa.load(wav_, sr=sr)
    #print(rate,sig)

    sig = preemphasis(sig, PREEMPH)
    frames = framesig(sig, winlen * rate, winstep * rate, HAMMING_WINFUNC)
    complex_spec = np.fft.rfft(frames, NFFT)

    time_scaled_complex_spec = None
    if with_time_scaled:
        time_scaled_frames = np.arange(frames.shape[-1]) * frames
        time_scaled_complex_spec = np.fft.rfft(time_scaled_frames, NFFT)

    print(complex_spec.shape, time_scaled_complex_spec.shape)
    return complex_spec, time_scaled_complex_spec


def get_mag_spec(complex_spec):
    """Return mag spec
    """
    return np.absolute(complex_spec)


def get_phase_spec(complex_spec):
    """Return phase spec
    """
    return np.angle(complex_spec)


def get_real_spec(complex_spec):
    """Return real spec
    """
    return np.real(complex_spec)


def get_imag_spec(complex_spec):
    """Return imag spec
    """
    return np.imag(complex_spec)


def cepstrally_smoothing(spec):
    """Return cepstrally smoothed spec
    """
    _spec = np.where(spec == 0, np.finfo(float).eps, spec)
    log_spec = np.log(_spec)
    ceps = np.fft.irfft(log_spec, NFFT)
    win = (np.arange(ceps.shape[-1]) < LIFTER).astype(np.float)
    win[LIFTER] = 0.5
    return np.absolute(np.fft.rfft(ceps * win, NFFT))


def get_modgdf(complex_spec, complex_spec_time_scaled):
    """Get Modified Group-Delay Feature
    """
    mag_spec = get_mag_spec(complex_spec)
    cepstrally_smoothed_mag_spec = cepstrally_smoothing(mag_spec)
    #plot_data(cepstrally_smoothed_mag_spec,"cepstrally_smoothed_mag_spec.png","cepstrally_smoothed_mag_spec")

    real_spec = get_real_spec(complex_spec)
    imag_spec = get_imag_spec(complex_spec)
    real_spec_time_scaled = get_real_spec(complex_spec_time_scaled)
    imag_spec_time_scaled = get_imag_spec(complex_spec_time_scaled)

    __divided = real_spec * real_spec_time_scaled \
            + imag_spec * imag_spec_time_scaled
    __tao = __divided / (cepstrally_smoothed_mag_spec ** (2. * GAMMA))
    __abs_tao = np.absolute(__tao)
    __sign = 2. * (__tao == __abs_tao).astype(np.float) - 1.
    return dct(__sign * (__abs_tao ** ALPHA), type=2, axis=1, norm='ortho')

def load_model(out_features, hidden_units, classifier_only=True):
    """ Load the pretrained model

    out_features    - number of features to predict
    classifier_only - change classifier parameters only
    """
    method_to_call = getattr(models, 'vgg16')
    model = method_to_call(pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

    # model = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
    # for param in model.parameters() :
    # param.requires_grad = False
    if classifier_only:
        # Do not update model parameters
        for param in model.parameters():
            param.requires_grad = False

    # Add our own classifier
    create_classifier(model, out_features=out_features,
                      hidden_units=hidden_units)

    return model
def readAudioFile(filename, sr=None):
    """Reads an audio file with default sampling rate 48000Hz
    filename - file to be read
    return - numpy.float32
    """

    x, sr = librosa.load(filename, sr=sr)
    return x, sr
class_to_idx = {'airport': 0, 'bus': 1, 'metro': 2, 'metro_station': 3, 'park': 4, 'public_square': 5, 'shopping_mall': 6, 'street_pedestrian': 7, 'street_traffic': 8, 'tram': 9}
def transform(path, sr):
    x, sr = readAudioFile(path)

    complex_spec, complex_spec_time_scaled = get_complex_spec(path, 0.079, 0.025, with_time_scaled=True)
    modgdf = get_modgdf(complex_spec, complex_spec_time_scaled)
    modgdf = np.absolute(modgdf)
    # print(modgdf.shape)
    # plot_data(modgdf, "modgdf.png", "modgdf")
    # plot_data(np.absolute(modgdf), "abs_modgdf.png", "abs_modgdf")

    hop_length = 1875  # This gives us 256 time buckets: 1875 = 10 * 48000 / 256
    n_fft = 8192  # This sets the lower frequency cut off to 48000 Hz / 8192 * 2 = 12 Hz
    S = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length)
    logS = librosa.power_to_db(abs(S))
    # return logS
    # print(modgdf)
    # print(logS.shape,modgdf.shape)

    img1 = Image.fromarray(logS)
    img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)

    img2 = Image.fromarray(modgdf)
    img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
    basepath = os.path.dirname(__file__)
    file_path1 = os.path.join(
        basepath, 'images', secure_filename("im1.png"))
    plt.imsave(file_path1, img1, cmap=plt.cm.gray)
    file_path2 = os.path.join(
        basepath, 'images', secure_filename("im2.png"))
    plt.imsave(file_path2, img2, cmap=plt.cm.gray)
    img1 = cv2.imread(file_path1)
    img2 = cv2.imread(file_path2)
    im_h = cv2.hconcat([img1, img2])
    transform_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    loader = transforms.Compose([transforms.ToTensor(), transform_norm, transforms.Resize([128, 256])])
    image = loader(im_h).float()
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    model = load_model(10, 512)
    file_pathm = os.path.join(basepath, secure_filename("mix-checkpoint-7.pt"))
    model.load_state_dict(torch.load(file_pathm, map_location=torch.device('cpu')))
    model.eval()
    out = {}
    with torch.no_grad():
        out_data = model(image)
        g = out_data.cpu().numpy().flatten()
        s = g.argsort()[-3:][::-1]
        v = 1
        for i in s:
            for d, k in class_to_idx.items():
                if k == i:
                    out[v] = d
                    v+= 1

        put = '1st location is '+out[3]+' \n '+ '2st location is '+out[2]+'  \n  3st location is '+out[1]

    return put
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        result = transform(file_path, sr = 48000)
        return result
    return None



if __name__=="__main__":
    app.run(debug=True)