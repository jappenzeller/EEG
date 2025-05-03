function render_channel_comparison_frame(data_folder, channel_idx, frame_number)
    % Render a single frame comparing 1 channel's ictal vs interictal EEG
    % Centered around t = 0.5s with 400ms spectrogram window to match 400Hz sampling

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
    n_fft = 160; % 400ms window at 400Hz sampling
    n_overlap = 120;
    window_sec = 1; % 1-second sliding window
    half_win = fs / 2;

    % Match segments
    ictal_files = dir(fullfile(data_folder, 'Dog_1_ictal_segment_*.mat'));
    interictal_files = dir(fullfile(data_folder, 'Dog_1_interictal_segment_*.mat'));
    get_index = @(name) sscanf(name, 'Dog_1_ictal_segment_%d.mat');
    ictal_indices = arrayfun(@(f) get_index(f.name), ictal_files);
    interictal_indices = arrayfun(@(f) sscanf(f.name, 'Dog_1_interictal_segment_%d.mat'), interictal_files);
    common_indices = intersect(ictal_indices, interictal_indices);
    common_indices = sort(common_indices);

    % Compute which segment and local frame index
    frames_per_segment = fs / step_size;
    seg_idx = floor((frame_number - 1) / frames_per_segment) + 1;
    local_frame = mod(frame_number - 1, frames_per_segment) * step_size + step_size;

    if seg_idx > length(common_indices)
        error('Frame number exceeds available segments.');
    end

    ictal = load_segment(data_folder, 'ictal', common_indices(seg_idx));
    interictal = load_segment(data_folder, 'interictal', common_indices(seg_idx));

    % Define center-based window indices
    total_samples = size(ictal, 2);
    center = local_frame;
    idx_range = max(1, center - half_win + 1) : min(total_samples, center + half_win);
    t = (1:length(idx_range)) / fs - 0.5; % center window at t = 0.5s

    % Axis range
    min_val = min([ictal(channel_idx,:), interictal(channel_idx,:)]) * 1.1;
    max_val = max([ictal(channel_idx,:), interictal(channel_idx,:)]) * 1.1;

    figure('Position', [100 100 1400 700]);

    % Interictal Amplitude
    subplot(2, 2, 1);
    plot(t, interictal(channel_idx, idx_range));
    title(sprintf('Interictal Amplitude (Ch %d)', channel_idx));
    ylim([min_val, max_val]); xlim([0, 1]);
    xlabel('Time (s)'); ylabel('Amplitude');

    % Interictal Spectrogram
    subplot(2, 2, 2);
    spectrogram(interictal(channel_idx, :), n_fft, n_overlap, n_fft, fs, 'yaxis');
    ylim([0, 60]);
    title('Interictal Spectrogram (400ms window)');

    % Ictal Amplitude
    subplot(2, 2, 3);
    plot(t, ictal(channel_idx, idx_range));
    title(sprintf('Ictal Amplitude (Ch %d)', channel_idx));
    ylim([min_val, max_val]); xlim([0, 1]);
    xlabel('Time (s)'); ylabel('Amplitude');

    % Ictal Spectrogram
    subplot(2, 2, 4);
    spectrogram(ictal(channel_idx, :), n_fft, n_overlap, n_fft, fs, 'yaxis');
    ylim([0, 60]);
    title('Ictal Spectrogram (400ms window)');

    sgtitle(sprintf('Segment %d - Channel %d Frame %d (centered at 0.5s)', common_indices(seg_idx), channel_idx, frame_number));
end
