function plot_ictal_segment_one(data_folder, channel_idx)
    % Plot line chart of Dog_1_ictal_segment_1.mat for a given channel
    % Default data_folder is 'H:\Data\PythonDNU\EEG\DataKaggle\'
    % Usage: plot_ictal_segment_one(); or plot_ictal_segment_one(folder, channel)

    if nargin < 1 || isempty(data_folder)
        data_folder = 'H:\Data\PythonDNU\EEG\DataKaggle\';
    end
    if nargin < 2
        channel_idx = 1;
    end

    fs = 400; % Sampling rate (Hz)
    fname = fullfile(data_folder, 'Dog_1', 'Dog_1_ictal_segment_1.mat');
    raw = load(fname);

    if isfield(raw, 'data')
        data = double(raw.data);
    else
        f = fieldnames(raw);
        data = double(raw.(f{1}).data);
    end

    % Extract channel signal and time vector
    signal = data(channel_idx, :);
    t = (0:length(signal)-1) / fs;

    % Plot
    figure;
    plot(t, signal, 'k');
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('Dog_1 Ictal Segment 1 - Channel %d', channel_idx));
    grid on;
end
