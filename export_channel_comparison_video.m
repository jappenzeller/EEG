function export_channel_comparison_video(data_folder, channel_idx, output_file, max_seconds)
    % Export a video comparing 1 channel's ictal vs interictal EEG over time
    % Each frame centered in a 1-second window with 40 fps playback

    if nargin < 1
        data_folder = uigetdir(pwd, 'Select Data Folder');
        if data_folder == 0, return; end
    end
    if nargin < 2
        channel_idx = 1;
    end
    if nargin < 3
        output_file = sprintf('channel%d_comparison.mp4', channel_idx);
    end
    if nargin < 4
        max_seconds = inf;
    end

    fs = 400;
    step_size = 10;
    frame_rate = 40;
    frames_per_segment = fs / step_size;

    % Match ictal/interictal segments
    ictal_files = dir(fullfile(data_folder, 'Dog_1_ictal_segment_*.mat'));
    interictal_files = dir(fullfile(data_folder, 'Dog_1_interictal_segment_*.mat'));
    get_index = @(name) sscanf(name, 'Dog_1_ictal_segment_%d.mat');
    ictal_indices = arrayfun(@(f) get_index(f.name), ictal_files);
    interictal_indices = arrayfun(@(f) sscanf(f.name, 'Dog_1_interictal_segment_%d.mat'), interictal_files);
    common_indices = intersect(ictal_indices, interictal_indices);
    common_indices = sort(common_indices);

    max_frames = min(length(common_indices) * frames_per_segment, max_seconds * frame_rate);

    hFig = figure('Visible', 'off', 'Position', [100 100 1400 700]);
    v = VideoWriter(output_file, 'MPEG-4');
    v.FrameRate = frame_rate;
    open(v);

    for frame_number = 1:max_frames
        % Reuse the frame rendering logic from the interactive version
        clf(hFig);
        render_channel_comparison_frame(data_folder, channel_idx, frame_number);
        drawnow;
        frame = getframe(hFig);
        writeVideo(v, frame);
    end

    close(v);
    close(hFig);
    fprintf('🎞️ Exported video with centered sliding window to %s\n', output_file);
end
