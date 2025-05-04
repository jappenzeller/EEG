function generate_dog1_segment_scroll_video(segment_indices, output_video)
    % Generates a scrolling side-by-side comparison video for Dog_1
    % segment_indices: array of segment indices to animate (e.g. 1:5)
    % output_video: optional .mp4 output path

    if nargin < 2
        output_video = 'Dog_1_segment_comparison_scroll.mp4';
    end

    base_path = 'H:\Data\PythonDNU\EEG\DataKaggle\Dog_1';
    hFig = figure('Visible', 'off', 'Position', [100, 100, 1200, 600]);
    v = VideoWriter(output_video, 'MPEG-4');
    v.FrameRate = 1; % One segment per second
    open(v);

    for idx = segment_indices
        ictal_file = fullfile(base_path, sprintf('Dog_1_ictal_segment_%d.mat', idx));
        interictal_file = fullfile(base_path, sprintf('Dog_1_interictal_segment_%d.mat', idx));

        if ~isfile(ictal_file) || ~isfile(interictal_file)
            fprintf('⚠️ Skipping segment %d (missing files)\n', idx);
            continue;
        end

        [ictal, t_ictal, fs] = load_segment(ictal_file);
        [interictal, t_inter, ~] = load_segment(interictal_file);

        % Clean frame
        clf(hFig);

        % Global Y-axis range
        all_data = [ictal.data(:); interictal.data(:)];
        y_min = min(all_data);
        y_max = max(all_data);

        % Find usable channels
        valid_channels = [];
        for i = 1:min(size(ictal.data,1), size(interictal.data,1))
            if any(ictal.data(i,:) ~= 0) && any(interictal.data(i,:) ~= 0)
                valid_channels(end+1) = i;
            end
        end
        n_plot = min(5, length(valid_channels));

        for j = 1:n_plot
            i = valid_channels(j);

            % Interictal (left)
            subplot(n_plot, 2, 2*(j-1)+1);
            plot(t_inter, interictal.data(i, :));
            title(sprintf('Interictal - Ch%d', i));
            ylim([y_min, y_max]);
            xlim([0, 1]);
            set(gca, 'Color', [0.8 1 0.8]);

            % Ictal (right)
            subplot(n_plot, 2, 2*j);
            plot(t_ictal, ictal.data(i, :));
            title(sprintf('Ictal - Ch%d', i));
            ylim([y_min, y_max]);
            xlim([0, 1]);
            set(gca, 'Color', [1 0.8 0.8]);
        end

        sgtitle(sprintf('Dog_1 Segment %d | 1-second EEG Comparison', idx));
        writeVideo(v, getframe(hFig));
    end

    close(v);
    close(hFig);
    fprintf('✅ Finished: %s\n', output_video);
end

function [seg_struct, time_vec, fs] = load_segment(path)
    raw = load(path);
    if isfield(raw, 'data')
        seg_struct.data = double(raw.data);
        fs = double(raw.freq);
    else
        keys = fieldnames(raw);
        content = raw.(keys{1});
        seg_struct.data = double(content.data);
        fs = double(content.freq);
    end
    time_vec = linspace(0, size(seg_struct.data, 2) / fs, size(seg_struct.data, 2));
end
