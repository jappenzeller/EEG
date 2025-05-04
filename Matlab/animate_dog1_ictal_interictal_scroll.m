function animate_dog1_ictal_interictal_scroll(data_folder)
    % Animate a 2-second sliding EEG window, side-by-side (interictal vs ictal)
    % Usage: animate_dog1_ictal_interictal_scroll('H:\Data\PythonDNU\EEG\DataKaggle\Dog_1')

    if nargin < 1
        data_folder = uigetdir(pwd, 'Select Dog_1 Folder');
        if data_folder == 0, return; end
    end

    % Load and match ictal/interictal segments
    ictal_files = dir(fullfile(data_folder, 'Dog_1_ictal_segment_*.mat'));
    interictal_files = dir(fullfile(data_folder, 'Dog_1_interictal_segment_*.mat'));

    get_index = @(name) sscanf(name, 'Dog_1_ictal_segment_%d.mat');
    ictal_indices = arrayfun(@(f) get_index(f.name), ictal_files);
    interictal_indices = arrayfun(@(f) sscanf(f.name, 'Dog_1_interictal_segment_%d.mat'), interictal_files);
    common_indices = intersect(ictal_indices, interictal_indices);
    common_indices = sort(common_indices);

    % Load all data
    ictal_data = concat_segments(data_folder, 'ictal', common_indices);
    interictal_data = concat_segments(data_folder, 'interictal', common_indices);

    [n_channels, total_samples] = size(ictal_data);
    fs = 400;
    win_sec = 2;
    win_samples = fs * win_sec;
    half_win = win_samples / 2;
    step_size = 20;
    frame_delay = step_size / fs;

    offset = max([abs(ictal_data(:)); abs(interictal_data(:))]) * 1.2;
    hFig = figure('Name', 'Dog_1 Sliding EEG Window', 'NumberTitle', 'off', 'Position', [100, 100, 1400, 600]);

    for center = half_win + 1 : step_size : total_samples - half_win
        idx = center - half_win + 1 : center + half_win;
        t = (1:win_samples) / fs;

        % --- Interictal subplot ---
        subplot(1, 2, 1);
        cla;
        hold on;
        for ch = 1:n_channels
            y_offset = (ch - 1) * offset;
            plot(t, interictal_data(ch, idx) + y_offset);
        end
        title('Interictal');
        xlim([0, win_sec]);
        ylim([-offset, n_channels * offset]);
        set(gca, 'YTick', (0:n_channels-1)*offset);
        set(gca, 'YTickLabel', arrayfun(@(x) sprintf('Ch%d', x), 1:n_channels, 'UniformOutput', false));
        set(gca, 'Color', [0.8 1 0.8]);
        xlabel('Time (s)');
        ylabel('Channels');

        % --- Ictal subplot ---
        subplot(1, 2, 2);
        cla;
        hold on;
        for ch = 1:n_channels
            y_offset = (ch - 1) * offset;
            plot(t, ictal_data(ch, idx) + y_offset);
        end
        title('Ictal');
        xlim([0, win_sec]);
        ylim([-offset, n_channels * offset]);
        set(gca, 'YTick', (0:n_channels-1)*offset);
        set(gca, 'YTickLabel', []);
        set(gca, 'Color', [1 0.8 0.8]);
        xlabel('Time (s)');

        % Super title and playback
        sgtitle(sprintf('Dog_1 EEG Sliding Window @ %.2f sec', center / fs));
        drawnow;
        pause(frame_delay);
    end

    fprintf('✅ Animation complete. Displayed ~%.2f seconds of EEG.\n', total_samples / fs);
end

function data = concat_segments(folder, label, indices)
    data = [];
    for i = 1:length(indices)
        fname = fullfile(folder, sprintf('Dog_1_%s_segment_%d.mat', label, indices(i)));
        raw = load(fname);
        if isfield(raw, 'data')
            segment = double(raw.data);
        else
            f = fieldnames(raw);
            segment = double(raw.(f{1}).data);
        end
        data = [data, segment];
    end
end
