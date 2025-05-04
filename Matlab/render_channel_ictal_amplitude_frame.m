function render_channel_ictal_amplitude_frame(data_folder, channel_idx, frame_number)
    % Render a single frame of ictal amplitude for one channel
    % Window is 1 second wide, with zero-padding if needed
    % Time starts at 0 on the left side of the graph

    if nargin < 1
        data_folder = uigetdir(pwd, 'Select Data Folder');
        if data_folder == 0, return; end
    end
    if nargin < 2
        channel_idx = 1;
    end
    if nargin < 3
        frame_number = 1;
    end

    % Setup
    fs = 400;
    step_size = 10;
    win_size = fs;     % 1-second window = 400 samples
    half_win = win_size / 2;

    % Match segments
    ictal_files = dir(fullfile(data_folder, 'Dog_1_ictal_segment_*.mat'));
    get_index = @(name) sscanf(name, 'Dog_1_ictal_segment_%d.mat');
    ictal_indices = arrayfun(@(f) get_index(f.name), ictal_files);
    ictal_indices = sort(ictal_indices);

    % Compute segment and local frame index
    frames_per_segment = fs / step_size;
    seg_idx = floor((frame_number - 1) / frames_per_segment) + 1;
    local_frame = mod(frame_number - 1, frames_per_segment) * step_size + step_size;

    if seg_idx > length(ictal_indices)
        error('Frame number exceeds available ictal segments.');
    end

    ictal = load_segment(data_folder, 'ictal', ictal_indices(seg_idx));
    signal = ictal(channel_idx, :);
    total_samples = length(signal);

    % Define centered window range with padding
    center = local_frame;
    left_idx = center - half_win + 1;
    right_idx = center + half_win;

    pad_left = max(0, 1 - left_idx);
    pad_right = max(0, right_idx - total_samples);

    valid_start = max(1, left_idx);
    valid_end = min(total_samples, right_idx);
    window_data = signal(valid_start:valid_end);
    window_data = [zeros(1, pad_left), window_data, zeros(1, pad_right)];
    t = (0:length(window_data)-1) / fs;  % start at 0s on the left

    % Axis range
    min_val = min(signal) * 1.1;
    max_val = max(signal) * 1.1;

    figure('Position', [100 100 800 400]);

    % Ictal Amplitude Only
    plot(t, window_data, 'k');
    title(sprintf('Ictal Amplitude (Ch %d)', channel_idx));
    ylim([min_val, max_val]); xlim([0, 1]);
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on;

    sgtitle(sprintf('Dog_1 Ictal Segment %d - Channel %d Frame %d (centered)', ictal_indices(seg_idx), channel_idx, frame_number));
end

function data = load_segment(folder, label, idx)
    fname = fullfile(folder, sprintf('Dog_1_%s_segment_%d.mat', label, idx));
    raw = load(fname);
    if isfield(raw, 'data')
        data = double(raw.data);
    else
        f = fieldnames(raw);
        data = double(raw.(f{1}).data);
    end
end
