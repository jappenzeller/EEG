function animate_comparison_video(preictal_path, interictal_path, window_sec, step_sec, output_video)
    % Create a side-by-side EEG video: interictal vs preictal
    if nargin < 3, window_sec = 10; end
    if nargin < 4, step_sec = 1; end
    if nargin < 5, output_video = 'eeg_comparison.mp4'; end

    % Load EEG segments
    [preictal, t_pre, fs] = load_segment(preictal_path);
    [interictal, t_inter, ~] = load_segment(interictal_path);

    % Setup parameters
    window_samples = round(window_sec * fs);
    step_samples = round(step_sec * fs);
    max_start = min(length(t_pre), length(t_inter)) - window_samples;

    % Prepare figure and video writer
    hFig = figure('Visible', 'off', 'Position', [100, 100, 1200, 600]);
    v = VideoWriter(output_video, 'MPEG-4');
    v.FrameRate = 5;
    open(v);

    % Determine global Y-axis range
    all_data = [preictal.data(:); interictal.data(:)];
    global_min = min(all_data);
    global_max = max(all_data);

    % Animation loop
    for start_idx = 1:step_samples:max_start
        clf(hFig);
        stop_idx = start_idx + window_samples - 1;

        t_win_pre = t_pre(start_idx:stop_idx);
        t_win_int = t_inter(start_idx:stop_idx);
        data_win_pre = preictal.data(:, start_idx:stop_idx);
        data_win_int = interictal.data(:, start_idx:stop_idx);

        % Auto-select valid channels
        valid_channels = [];
        for i = 1:min(size(data_win_pre,1), size(data_win_int,1))
            pre = data_win_pre(i, :);
            inter = data_win_int(i, :);
            if any(pre ~= 0 & ~isnan(pre)) && any(inter ~= 0 & ~isnan(inter))
                valid_channels(end+1) = i;
            end
        end

        n_plot = min(5, length(valid_channels));
        for j = 1:n_plot
            i = valid_channels(j);
            pre = data_win_pre(i, :);
            inter = data_win_int(i, :);

            % Interictal (left)
            subplot(n_plot, 2, 2*(j-1)+1);
            plot(t_win_int, inter);
            title(sprintf('Interictal - %s', interictal.channels{i}));
            ylabel('Amplitude');
            ylim([global_min global_max]);
            xlim([t_win_int(1), t_win_int(end)]);
            set(gca, 'Color', [0.8 1 0.8]);

            % Preictal (right)
            subplot(n_plot, 2, 2*j);
            plot(t_win_pre, pre);
            title(sprintf('Preictal - %s', preictal.channels{i}));
            ylim([global_min global_max]);
            xlim([t_win_pre(1), t_win_pre(end)]);
            set(gca, 'Color', [1 0.8 0.8]);
        end

        sgtitle(sprintf('EEG Comparison | Time %.1f to %.1f sec', t_win_pre(1), t_win_pre(end)));
        writeVideo(v, getframe(hFig));
    end

    close(v);
    close(hFig);
    fprintf('✅ Video exported: %s\n', output_video);
end

% Helper: Load EEG segment
function [seg_struct, time_vec, fs] = load_segment(path)
    raw = load(path);
    top_field = fieldnames(raw);
    seg = raw.(top_field{1});
    seg_struct.data = double(seg.data);
    seg_struct.channels = cellfun(@char, seg.channels, 'UniformOutput', false);
    fs = double(seg.sampling_frequency);
    duration = double(seg.data_length_sec);
    time_vec = linspace(0, duration, size(seg.data, 2));
end
