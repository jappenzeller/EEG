function animate_ictal_lines(data_folder)
    % Animate line plots for each EEG channel across Dog_1 ictal segments
    % Usage: animate_ictal_lines('H:\Data\PythonDNU\EEG\DataKaggle\Dog_1')

    if nargin < 1
        data_folder = uigetdir(pwd, 'Select Dog_1 Folder');
        if data_folder == 0, return; end
    end

    % Get all ictal segment files and sort by index
    files = dir(fullfile(data_folder, 'Dog_1_ictal_segment_*.mat'));
    indices = arrayfun(@(f) sscanf(f.name, 'Dog_1_ictal_segment_%d.mat'), files);
    [~, sorted_idx] = sort(indices);
    files = files(sorted_idx);

    % Setup figure
    hFig = figure('Name', 'Dog_1 Ictal Segment Line Animation', 'NumberTitle', 'off');

    for k = 1:length(files)
        file = fullfile(data_folder, files(k).name);
        raw = load(file);

        % Load EEG data (flat or nested)
        if isfield(raw, 'data')
            eeg_data = double(raw.data);
        else
            f = fieldnames(raw);
            eeg_data = double(raw.(f{1}).data);
        end

        % Plot line for each channel
        clf(hFig);
        [n_channels, n_samples] = size(eeg_data);
        offset = max(abs(eeg_data(:))) * 1.2;

        hold on;
        for ch = 1:n_channels
            plot((1:n_samples), eeg_data(ch, :) + (ch-1)*offset);
        end
        hold off;

        yticks((0:n_channels-1)*offset);
        yticklabels(arrayfun(@(x) sprintf('Ch%d', x), 1:n_channels, 'UniformOutput', false));
        xlabel('Time (samples)');
        ylabel('Channels (stacked)');
        title(sprintf('Dog_1 Ictal Segment %d', indices(sorted_idx(k))));
        axis tight;

        pause(1);  % Pause 1 second per frame
    end

    fprintf('✅ Line animation complete. Displayed %d segments.\n', length(files));
end
