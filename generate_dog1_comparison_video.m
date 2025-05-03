function generate_dog1_comparison_video(segment_index, output_video)
    % Create a Dog_1 ictal vs interictal comparison video from Kaggle dataset
    % segment_index: index to match ictal and interictal clips (e.g. 1)
    % output_video: optional name for exported .mp4

    if nargin < 2
        output_video = sprintf('Dog_1_segment_%04d_comparison.mp4', segment_index);
    end

    % Define path
    base_path = 'H:\Data\PythonDNU\EEG\DataKaggle\Dog_1';
    seg_str = sprintf('%d', segment_index);

    % Construct filenames
    ictal_file = fullfile(base_path, sprintf('Dog_1_ictal_segment_%s.mat', seg_str));
    interictal_file = fullfile(base_path, sprintf('Dog_1_interictal_segment_%s.mat', seg_str));

    if ~isfile(ictal_file) || ~isfile(interictal_file)
        error('Segment %d not found in both ictal and interictal forms.', segment_index);
    end

    % Load both segments
    [ictal, t_ictal, fs] = load_segment(ictal_file);
    [interictal, t_inter, ~] = load_segment(interictal_file);

    % Set up video
    hFig = figure('Visible', 'off', 'Position', [100, 100, 1200, 600]);
    v = VideoWriter(output_video, 'MPEG-4');
    v.FrameRate = 1;  % One frame per second segment
    open(v);

    % Plot one frame (1 second)
    clf(hFig);
    data_ictal = ictal.data;
    data_inter = interictal.data;

    % Global y-axis
    all_data = [data_ictal(:); data_inter(:)];
    global_min = min(all_data);
    global_max = max(all_data);

    % Find valid channels
    valid_channels = [];
    for i = 1:min(size(data_ictal,1), size(data_inter,1))
        if any(data_ictal(i,:) ~= 0) && any(data_inter(i,:) ~= 0)
            valid_channels(end+1) = i;
        end
    end
    n_plot = min(5, length(valid_channels));

    for j = 1:n_plot
        i = valid_channels(j);

        % Interictal
        subplot(n_plot, 2, 2*(j-1)+1);
        plot(t_inter, data_inter(i, :));
        title(sprintf('Interictal - Ch%d', i));
        ylim([global_min global_max]);
        xlim([t_inter(1), t_inter(end)]);
        set(gca, 'Color', [0.8 1 0.8]);

        % Ictal
        subplot(n_plot, 2, 2*j);
        plot(t_ictal, data_ictal(i, :));
        title(sprintf('Ictal - Ch%d', i));
        ylim([global_min global_max]);
        xlim([t_ictal(1), t_ictal(end)]);
        set(gca, 'Color', [1 0.8 0.8]);
    end

    sgtitle(sprintf('Dog_1 EEG Comparison | Segment %d', segment_index));
    writeVideo(v, getframe(hFig));
    close(v);
    close(hFig);

    fprintf('✅ Exported video: %s\n', output_video);
end

function [seg_struct, time_vec, fs] = load_segment(path)
    raw = load(path);                   % Load the .mat file
    keys = fieldnames(raw);            % There should only be one variable inside
    content = raw.(keys{1});           % Get the actual struct

    if isstruct(content) && isfield(content, 'data')
        % The data is nested inside a named struct
        seg_struct.data = double(content.data);
        fs = double(content.freq);
        time_vec = linspace(0, size(content.data, 2) / fs, size(content.data, 2));
    else
        % Data is flat in top-level fields
        seg_struct.data = double(raw.data);
        fs = double(raw.freq);
        time_vec = linspace(0, size(raw.data, 2) / fs, size(raw.data, 2));
    end
end

