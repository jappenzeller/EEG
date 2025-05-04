function animate_preictal_segment_video(filepath, window_sec, step_sec, output_video)
    % Animate and export a scrolling preictal EEG segment to video
    % filepath      - path to .mat file
    % window_sec    - window duration in seconds (e.g. 10)
    % step_sec      - step between frames (e.g. 1)
    % output_video  - name of the output .mp4 file

    if nargin < 1
        [file, path] = uigetfile('*.mat', 'Select a Preictal EEG Segment');
        if isequal(file, 0), return; end
        filepath = fullfile(path, file);
    end
    if nargin < 2, window_sec = 10; end
    if nargin < 3, step_sec = 1; end
    if nargin < 4, output_video = 'preictal_eeg_animation.mp4'; end

    % Load data
    mat_struct = load(filepath);
    top_field = fieldnames(mat_struct);
    seg = mat_struct.(top_field{1});
    data = double(seg.data);
    fs = double(seg.sampling_frequency);
    duration = double(seg.data_length_sec);
    t = linspace(0, duration, size(data, 2));

    [num_channels, total_samples] = size(data);
    window_samples = round(window_sec * fs);
    step_samples = round(step_sec * fs);
    max_start = total_samples - window_samples;

    % Channel labels
    if iscell(seg.channels)
        ch = seg.channels;
    else
        ch = cellfun(@char, seg.channels, 'UniformOutput', false);
    end

    % Set up video writer
    v = VideoWriter(output_video, 'MPEG-4');
    v.FrameRate = 5;  % slow scroll
    open(v);

    % Create figure (offscreen)
    hFig = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);

    n_plot = min(5, num_channels);
    for start_idx = 1:step_samples:max_start
        clf(hFig);
        stop_idx = start_idx + window_samples - 1;
        t_window = t(start_idx:stop_idx);
        data_window = data(1:n_plot, start_idx:stop_idx);

        for i = 1:n_plot
            subplot(n_plot, 1, i);
            plot(t_window, data_window(i, :));
            ylabel(ch{i}, 'FontSize', 8);
            if i == 1
                title(sprintf('Preictal EEG - Time %.1f to %.1f sec', ...
                    t_window(1), t_window(end)));
            end
            axis tight;
        end
        xlabel('Time (s)');
        frame = getframe(hFig);
        writeVideo(v, frame);
    end

    close(v);
    close(hFig);

    fprintf('✅ Video exported to: %s\n', output_video);
end
