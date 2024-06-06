function get_spectrogram(x, fs)
    window_length = round(0.03 * fs);  % 30 ms 窗长
    overlap = round(0.006 * fs);      % 6 ms 重叠
    nfft = 2^nextpow2(window_length);  % FFT 点数

    [S, F, T, P] = melSpectrogram(x, fs);

    P_dB = 10*log10(abs(P));
    % 自定义绘制频谱图
    figure;
    imagesc(T, F, P_dB);
    axis xy; % 确保 y 轴从下到上增加
    colormap(jet); % 使用 jet 颜色图
    colorbar; % 显示颜色条
    title('Spectrogram (Power Spectral Density)');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
end
