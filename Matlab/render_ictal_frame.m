function render_ictal_frame(data_folder, subject, channels, frame_number)
    % Render a single frame from ictal EEG data with amplitude and waterfall plots
    % Includes UI arrows to navigate forward/backward through frames

    if nargin < 1 || isempty(data_folder)
        data_folder = 'H:\Data\PythonDNU\EEG\DataKaggle';
    end
    if nargin < 2 || isempty(subject)
        subject = 'Dog_1';
    end
    if nargin < 3 || isempty(channels)
        channels = 1:16;
    end
    if nargin < 4 || isempty(frame_number)
        frame_number = 1;
    end

    fs = 400;
    win_size = fs;  % 1-second window
    step_size = 10;
    all_signal = [];

    subject_folder = fullfile(data_folder, subject);
    ictal_files = dir(fullfile(subject_folder, sprintf('%s_ictal_segment_*.mat', subject)));
    get_segment = @(f) load_segment(fullfile(subject_folder, f.name));

    for k = 1:length(ictal_files)
        seg = get_segment(ictal_files(k));
        all_signal = [all_signal, seg(channels, :)];
    end

    total_samples = size(all_signal, 2);
    num_channels = length(channels);
    max_frames = floor((total_samples - win_size + 1) / step_size);

    fig = figure('Position', [100, 100, 1000, 400 * num_channels], 'KeyPressFcn', @key_callback);
    render_frame(frame_number);

    uicontrol('Style', 'pushbutton', 'String', '<', 'Position', [20, 20, 50, 30], 'Callback', @back_callback);
    uicontrol('Style', 'pushbutton', 'String', '>', 'Position', [80, 20, 50, 30], 'Callback', @forward_callback);

    
    

        function render_frame(frame_idx)
        frame_number = frame_idx;
        delete(findall(fig, 'Type', 'axes'));
        i = (frame_idx - 1) * step_size + 1;
        if i + win_size - 1 > total_samples
            return;
        end

        t_start = (i - 1) / fs;
        t_window = t_start + (0:win_size - 1) / fs;

        sgtitle(sprintf('%s Ictal Frame %d (t = %.2fs)', subject, frame_idx, t_start));

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
        

    

end

    function back_callback(~, ~)
        render_frame(max(frame_number - 1, 1));
    end

    function forward_callback(~, ~)
        render_frame(min(frame_number + 1, max_frames));
    end

    function key_callback(~, event)
        switch event.Key
            case 'period'
                frame_number = min(frame_number + 1, max_frames);
            case 'comma'
                frame_number = max(frame_number - 1, 1);
        end
        render_frame(frame_number);
    end
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
