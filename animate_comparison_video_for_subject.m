function animate_comparison_video_for_subject(subject_root, subject_id, segment_idx, output_video)
    % Generates a side-by-side interictal vs preictal EEG video for a subject and index
    %
    % subject_root: root data folder (e.g. 'H:\Data\PythonDNU\EEG\Data')
    % subject_id: folder name like 'dog_1'
    % segment_idx: numeric index of the file (e.g., 1)
    % output_video: optional .mp4 file name

    if nargin < 4
        output_video = sprintf('%s_segment_%04d_comparison.mp4', upper(subject_id), segment_idx);
    end

    subfolder = fullfile(subject_root, subject_id, subject_id);

    % Generate file names
    seg_str = sprintf('%04d', segment_idx);
    preictal_file = fullfile(subfolder, sprintf('%s_preictal_segment_%s.mat', capitalize(subject_id), seg_str));
    interictal_file = fullfile(subfolder, sprintf('%s_interictal_segment_%s.mat', capitalize(subject_id), seg_str));

    if ~isfile(preictal_file)
        error('Preictal file not found: %s', preictal_file);
    end
    if ~isfile(interictal_file)
        error('Interictal file not found: %s', interictal_file);
    end

    % Call the base comparison function
    animate_comparison_video(preictal_file, interictal_file, 10, 1, output_video);
end

function str = capitalize(s)
    str = lower(s);
    str(1) = upper(str(1));
end
