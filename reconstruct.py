import os
import numpy as np
import scipy.io
import librosa
import soundfile as sf

os.makedirs('output', exist_ok=True)

y_pred = scipy.io.loadmat('output.mat')['output']
mels = []
for i in range(y_pred.shape[0]):
    mel = y_pred[i].reshape(1024, 32)
    mels.append(mel)
mels = np.array(mels)
audio = librosa.feature.inverse.mel_to_audio(mels, sr=16000)

for i in range(audio.shape[0]):
    sf.write(f'{i}.mp3', audio[i], 16000)
