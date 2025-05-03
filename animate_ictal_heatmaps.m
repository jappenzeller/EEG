function animate_ictal_heatmaps(data_folder)
    % Animates heatmaps for sequential Dog_1 ictal EEG segments
    % Example: animate_ictal_heatmaps('H:\Data\PythonDNU\EEG\DataKaggle\Dog_1')

    if nargin < 1
        data_folder = uigetdir(pwd, 'Select Dog_1 Folder');
        if data_folder == 0, return; end
    end

    % Get all ictal segment files
    files = dir(fullfile(data_folder, 'Dog_1_ictal_segment_*.mat'));

    % Sort files numerically by segment index
    indices = arrayfun(@(f) sscanf(f.name, 'Dog_1_ictal_segment_%d.mat'), files);
    [~, sorted_idx] = sort(indices);
    files = files(sorted_idx);

    % Create figure
    hFig = figure('Name', 'Dog_1 Ictal Segment Heatmap Animation', 'NumberTitle', 'off');

    for k = 1:length(files)
        file = fullfile(data_folder, files(k).name);
        raw = load(file);

        % Support both flat and nested .mat files
        if isfield(raw, 'data')
            eeg_data = double(raw.data);
        else
            field = fieldnames(raw);
            eeg_data = double(raw.(field{1}).data);
        end

        % Plot heatmap using imagesc for speed
        clf(hFig);
        imagesc(eeg_data);
        colormap(jet);
        colorbar;
        title(sprintf('Dog_1 Ictal Segment %d', indices(sorted_idx(k))));
        xlabel('Time (samples)');
        ylabel('Channel');
        set(gca, 'YDir', 'normal');  % Channel 1 at top

        pause(1);  % 1 second per segment
    end

    fprintf('✅ Animation complete. Displayed %d segments.\n', length(files));
end
