# voice-conversion
数字信号处理课程大作业：基于深度学习的人声转换器

# 系统结构
- 数据处理层：划分训练集、测试集；去噪，并在原始音频上应用汉明窗以削减频谱泄漏
- 特征编码层：对音频进行短时傅里叶变换，构造梅尔频谱特征
- 特征映射层：将梅尔频谱特征送入LSTM+全连接层网络，映射为目标音频特征
- 音频重建层：将映射后的特征重构为音频信号

# 部署方式
- 部署环境：Python 3.12、MATLAB R2024a
- 执行 `pip install -r requirements.txt` 安装必要依赖库
- 将 `data.zip` 解压至 `./data/train/` 文件夹下
- 用MATLAB运行 `generate_data.m` 文件，通过原始音频构造梅尔频谱图特征；此时目录下会生成 `train_data.mat` 文件
- 用Python运行 `train.py` 文件，模型开始训练，训练完毕后自动将音频特征保存至 `output.mat` 文件
- 运行 `reconstruct.py` ，将输出特征重构为音频文件，结果将输出至 `output` 文件夹
