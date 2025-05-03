function generate_dog1_comparison_videos_batch()
    % Generate ictal vs interictal videos for all matching segments in Dog_1
    base_path = 'H:\Data\PythonDNU\EEG\DataKaggle\Dog_1';
    listing = dir(fullfile(base_path, 'Dog_1_ictal_segment_*.mat'));

    % Extract all segment indices that have matching interictal pairs
    segment_indices = [];
    for k = 1:length(listing)
        name = listing(k).name;
        token = regexp(name, 'Dog_1_ictal_segment_(\d+)\.mat', 'tokens');
        if ~isempty(token)
            idx = str2double(token{1});
            interictal_file = fullfile(base_path, sprintf('Dog_1_interictal_segment_%d.mat', idx));
            if isfile(interictal_file)
                segment_indices(end+1) = idx;
            end
        end
    end

    fprintf('🧠 Found %d matching ictal/interictal pairs.\n', numel(segment_indices));

    % Loop through all matched pairs and generate video
    for i = 1:numel(segment_indices)
        idx = segment_indices(i);
        output_file = sprintf('Dog_1_segment_%04d_comparison.mp4', idx);
        fprintf('📹 Generating video for segment %d...\n', idx);
        try
            generate_dog1_comparison_video(idx, output_file);
        catch ME
            warning('⚠️ Failed to generate video for segment %d: %s', idx, ME.message);
        end
    end

    fprintf('✅ All videos generated.\n');
end
