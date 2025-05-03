function animate_ictal_sliding_window(data_folder, subject, output_file, channels, max_seconds)
    % Animate a 1-second sliding window over selected channels of subject ictal segments
    % Each channel shows amplitude and waterfall plot (3D spectrogram), saved as an mp4 video (no display)

    if nargin < 1 || isempty(data_folder)
        data_folder = 'H:\Data\PythonDNU\EEG\DataKaggle';
    end
    if nargin < 2 || isempty(subject)
        subject = 'Dog_1';
    end
    if nargin < 3 || isempty(output_file)
        output_file = sprintf('%s_ictal_selected_channels.mp4', lower(subject));
    end
    if nargin < 4 || isempty(channels)
        channels = 1:16;
    end
    if nargin < 5 || isempty(max_seconds)
        max_seconds = inf;
    end

    fs = 400;  % Hz
    win_size = fs;  % 1-second window = 400 samples
    step_size = 10; % Step forward 25 ms at a time
    all_signal = [];

    subject_folder = fullfile(data_folder, subject);
    ictal_files = dir(fullfile(subject_folder, sprintf('%s_ictal_segment_*.mat', subject)));
    get_segment = @(f) load_segment(fullfile(subject_folder, f.name));

    for k = 1:length(ictal_files)
        seg = get_segment(ictal_files(k));
        all_signal = [all_signal, seg(channels, :)];
    end

    total_samples = size(all_signal, 2);
    max_samples = min(total_samples, floor(max_seconds * fs));
    num_channels = length(channels);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 400 * num_channels]);
    sg = sgtitle(sprintf('%s Ictal Sliding Window - Selected Channels', subject));

    v = VideoWriter(output_file, 'MPEG-4');
    v.FrameRate = fs / step_size;
    open(v);

    for i = 1:step_size:(max_samples - win_size + 1)
        t_start = (i - 1) / fs;
        t_window = t_start + (0:win_size - 1) / fs;

        for idx = 1:num_channels
            win_signal = all_signal(idx, i:i + win_size - 1);

            subplot(num_channels, 2, (idx - 1) * 2 + 1);
            plot(t_window, win_signal, 'k');
            title(sprintf('Ch %d - Amplitude', channels(idx)));
            set(gca, 'XTick', [], 'YTick', []);
            ylim([min(all_signal(idx, :)) * 1.1, max(all_signal(idx, :)) * 1.1]);
            xlim([t_window(1), t_window(end)]);

            subplot(num_channels, 2, (idx - 1) * 2 + 2);
            [S,F,T] = spectrogram(win_signal, 128, 120, 128, fs);
            surf(T + t_window(1), F, 10*log10(abs(S)), 'EdgeColor', 'none');
            view([0 90]);
            axis tight;
            ylim([0 60]);
            colormap(jet);
            title('Waterfall Plot');
            set(gca, 'XTick', [], 'YTick', []);
        end

        sg.String = sprintf('%s Ictal Sliding Window - Selected Channels (t = %.2fs)', subject, t_start);
        drawnow;
        frame = getframe(fig);
        writeVideo(v, frame);
    end

    close(v);
    close(fig);
    fprintf('🎞️ Exported video to %s\n', output_file);
end

function data = load_segment(fname)
    raw = load(fname);
    if isfield(raw, 'data')
        data = double(raw.data);
    else
        f = fieldnames(raw);
        data = double(raw.(f{1}).data);
    end
end