function frames = framesig(x, fs, frame_len, frame_step, win_func)
    frame_len = round(frame_len * fs / 1000);
    frame_step = round(frame_step * fs / 1000);
    audio_len = length(x);

    if frame_len >= audio_len
        n_frames = 1;
    else
        n_frames = floor((audio_len - frame_len) / frame_step) + 1;
    end

    win = win_func(n_frames)';
    frames = zeros(frame_len, n_frames);

    for i = 1:n_frames
        start_idx = (i-1) * frame_step + 1;
        end_idx = start_idx + frame_len - 1;
        frames(:, i) = x(start_idx : end_idx);
    end

    frames = frames .* win;
end
