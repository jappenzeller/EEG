function eeg_comparison_viewer_with_navigation(data_folder)
    % EEG viewer with comparison across Preictal, Interictal, Test with navigation
    if nargin < 1
        data_folder = uigetdir(pwd, 'Select EEG Data Folder');
        if data_folder == 0, return; end
    end

    files = dir(fullfile(data_folder, '*.mat'));
    if isempty(files), error('No .mat files found.'); end

    % Group files by type
    preictal_files = get_files_by_type(files, 'preictal');
    interictal_files = get_files_by_type(files, 'interictal');
    test_files = get_files_by_type(files, 'test');

    max_idx = max([numel(preictal_files), numel(interictal_files), numel(test_files)]);
    idx = 1;

    % Create figure
    hFig = figure('Name', 'EEG Comparison Viewer', ...
        'NumberTitle', 'off', 'KeyPressFcn', @key_callback);

    update_plot();

    function update_plot()
        clf(hFig);
        segments = {};
        labels = {};
        colors = {};
        filenames = {};

        % Load one of each, if available
        if idx <= numel(preictal_files)
            [seg, name] = load_segment(fullfile(data_folder, preictal_files(idx).name));
            segments{end+1} = seg; labels{end+1} = 'PREICTAL'; colors{end+1} = [1 0.8 0.8]; filenames{end+1} = name;
        end
        if idx <= numel(interictal_files)
            [seg, name] = load_segment(fullfile(data_folder, interictal_files(idx).name));
            segments{end+1} = seg; labels{end+1} = 'INTERICTAL'; colors{end+1} = [0.8 1 0.8]; filenames{end+1} = name;
        end
        if idx <= numel(test_files)
            [seg, name] = load_segment(fullfile(data_folder, test_files(idx).name));
            segments{end+1} = seg; labels{end+1} = 'TEST'; colors{end+1} = [0.8 0.8 1]; filenames{end+1} = name;
        end

        % Shared Y-range
        global_min = inf;
        global_max = -inf;
        for i = 1:length(segments)
            data = segments{i}.data;
            data = double(data(1:min(5, end), :));
            global_min = min(global_min, min(data(:)));
            global_max = max(global_max, max(data(:)));
        end

        for i = 1:length(segments)
            subplot(1, 3, i);
            plot_segment(segments{i}, labels{i}, colors{i}, filenames{i}, global_min, global_max);
        end

        sgtitle(sprintf('Segment Set %d of %d', idx, max_idx));
    end

    function key_callback(~, event)
        switch event.Key
            case 'rightarrow'
                idx = min(idx + 1, max_idx);
                update_plot();
            case 'leftarrow'
                idx = max(idx - 1, 1);
                update_plot();
        end
    end

    % --------- Helpers ----------
    function list = get_files_by_type(file_list, keyword)
        list = file_list(contains({file_list.name}, keyword, 'IgnoreCase', true));
    end

    function [seg, name] = load_segment(fullpath)
        mat_struct = load(fullpath);
        top_field = fieldnames(mat_struct);
        seg = mat_struct.(top_field{1});
        [~, name, ext] = fileparts(fullpath);
        name = [name ext];
    end

    function plot_segment(seg, label, color, fname, y_min, y_max)
        data = seg.data;
        duration = double(seg.data_length_sec);
        t = linspace(0, duration, size(data, 2));

        if iscell(seg.channels)
            ch = seg.channels;
        else
            ch = cellfun(@char, seg.channels, 'UniformOutput', false);
        end

        n_plot = min(5, size(data, 1));
        hold on;
        for i = 1:n_plot
            offset = (i - 1) * (y_max - y_min + 100);
            plot(t, double(data(i, :)) + offset);
        end
        title(sprintf('%s\n%s', label, fname), 'Interpreter', 'none');
        xlabel('Time (s)');
        ylabel('Amplitude + Offset');
        set(gca, 'Color', color);
        ylim([y_min, y_max] + [0, (n_plot - 1) * (y_max - y_min + 100)]);
        xlim([0, duration]);
        axis tight;
    end
end
