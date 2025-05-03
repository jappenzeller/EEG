function animate_preictal_segment(filepath, window_sec, step_sec)
    % Animate a preictal EEG segment scrolling over time
    % filepath  - full path to preictal .mat file
    % window_sec - window size in seconds (e.g. 10)
    % step_sec   - step size between frames in seconds (e.g. 1)

    if nargin < 1
        [file, path] = uigetfile('*.mat', 'Select a Preictal EEG Segment');
        if isequal(file, 0), return; end
        filepath = fullfile(path, file);
    end
    if nargin < 2, window_sec = 10; end
    if nargin < 3, step_sec = 1; end

    % Load and extract segment
    mat_struct = load(filepath);
    top_field = fieldnames(mat_struct);
    seg = mat_struct.(top_field{1});
    data = double(seg.data);  % channels x time
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

    % Setup figure
    n_plot = min(5, num_channels);
    hFig = figure('Name', 'Preictal EEG Animation', 'NumberTitle', 'off');
    
    % Animation loop
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
                title(sprintf('Time: %.1f to %.1f sec', t_window(1), t_window(end)));
            end
        end
        xlabel('Time (s)');
        drawnow;
    end
end
