import os
import numpy as np
import scipy.io
import librosa
import soundfile as sf
from tqdm import tqdm

print("开始处理数据...")
os.makedirs('output', exist_ok=True)

# 加载预测数据
print("正在加载.mat文件...")
y_pred = scipy.io.loadmat('output.mat')['output']
print("数据加载成功，维度:", y_pred.shape)

# 处理Mel频谱
print("正在处理Mel频谱...")
mels = []
for i in tqdm(range(y_pred.shape[0]), desc="处理Mel频谱", unit="frame"):
    mel = y_pred[i].reshape(1024, 32)
    mels.append(mel)
mels = np.array(mels)

# 转换为音频
print("正在将Mel频谱转换为音频...")
audio = librosa.feature.inverse.mel_to_audio(mels, sr=16000)
print("音频转换完成，生成音频段数量:", audio.shape[0])

# 保存音频文件
print("正在保存音频文件...")
for i in tqdm(range(audio.shape[0]), desc="保存音频", unit="file"):
    sf.write(os.path.join('output', f'{i}.mp3'), audio[i], 16000)

print(f"处理完成！共生成 {audio.shape[0]} 个音频文件，已保存到 output/ 目录")
