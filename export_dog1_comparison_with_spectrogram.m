function export_dog1_comparison_with_spectrogram(data_folder, output_file)
    % Export EEG comparison (interictal vs ictal) with spectrograms for each channel
    % Usage: export_dog1_comparison_with_spectrogram('path/to/Dog_1', 'comparison_spectrogram.mp4')

    if nargin < 1
        data_folder = uigetdir(pwd, 'Select Dog_1 Folder');
        if data_folder == 0, return; end
    end
    if nargin < 2
        output_file = 'dog1_comparison_with_spectrogram.mp4';
    end

    % Load and match segments
    ictal_files = dir(fullfile(data_folder, 'Dog_1_ictal_segment_*.mat'));
    interictal_files = dir(fullfile(data_folder, 'Dog_1_interictal_segment_*.mat'));
    get_index = @(name) sscanf(name, 'Dog_1_ictal_segment_%d.mat');
    ictal_indices = arrayfun(@(f) get_index(f.name), ictal_files);
    interictal_indices = arrayfun(@(f) sscanf(f.name, 'Dog_1_interictal_segment_%d.mat'), interictal_files);
    common_indices = intersect(ictal_indices, interictal_indices);
    common_indices = sort(common_indices);

    % Load and concatenate
    ictal_data = concat_segments(data_folder, 'ictal', common_indices);
    interictal_data = concat_segments(data_folder, 'interictal', common_indices);

    [n_channels, total_samples] = size(ictal_data);
    fs = 400;
    win_sec = 2;
    win_samples = fs * win_sec;
    half_win = win_samples / 2;
    step_size = 20;
    frame_delay = step_size / fs;

    % Set up figure and video writer
    hFig = figure('Visible', 'off', 'Position', [100 100 1600 900]);
    v = VideoWriter(output_file, 'MPEG-4');
    v.FrameRate = round(1 / frame_delay);
    open(v);

    for center = half_win + 1 : step_size : total_samples - half_win
        idx = center - half_win + 1 : center + half_win;
        t = (1:win_samples) / fs;

        clf(hFig);

        for ch = 1:min(4, n_channels)  % Display only 4 channels for clarity
            y_offset = (ch - 1) * 2 * fs;

            % --- Interictal waveform ---
            subplot(4, 4, (ch-1)*4 + 1);
            plot(t, interictal_data(ch, idx));
            title(sprintf('Interictal Ch%d', ch));
            xlim([0, win_sec]);
            ylim auto;
            xlabel('Time (s)'); ylabel('Amplitude');
            set(gca, 'Color', [0.8 1 0.8]);

            % --- Interictal spectrogram ---
            subplot(4, 4, (ch-1)*4 + 2);
            spectrogram(interictal_data(ch, idx), 64, 60, 128, fs, 'yaxis');
            title('Interictal Spectrogram');
            ylim([0 60]);

            % --- Ictal waveform ---
            subplot(4, 4, (ch-1)*4 + 3);
            plot(t, ictal_data(ch, idx));
            title(sprintf('Ictal Ch%d', ch));
            xlim([0, win_sec]);
            ylim auto;
            xlabel('Time (s)'); ylabel('Amplitude');
            set(gca, 'Color', [1 0.8 0.8]);

            % --- Ictal spectrogram ---
            subplot(4, 4, (ch-1)*4 + 4);
            spectrogram(ictal_data(ch, idx), 64, 60, 128, fs, 'yaxis');
            title('Ictal Spectrogram');
            ylim([0 60]);
        end

        sgtitle(sprintf('Dog_1 EEG + Spectrogram @ %.2f sec', center/fs));
        drawnow;
        frame = getframe(hFig);
        writeVideo(v, frame);
    end

    close(v);
    close(hFig);
    fprintf('🎥 Spectrogram video export complete: %s\n', output_file);
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