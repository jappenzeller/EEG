function windows = segment_into_windows(data, fs, win_sec, step_sec)
    % Slice continuous EEG data into overlapping windows
    % Input:
    %   data: [channels x total_time]
    %   fs: sampling frequency
    %   win_sec: window size in seconds
    %   step_sec: step size in seconds
    % Output:
    %   windows: 3D array [channels x win_samples x num_windows]

    [num_channels, total_samples] = size(data);
    win_samples = round(win_sec * fs);
    step_samples = round(step_sec * fs);

    num_windows = floor((total_samples - win_samples) / step_samples) + 1;
    windows = zeros(num_channels, win_samples, num_windows);

    for i = 1:num_windows
        start_idx = (i - 1) * step_samples + 1;
        end_idx = start_idx + win_samples - 1;
        windows(:, :, i) = data(:, start_idx:end_idx);
    end
end