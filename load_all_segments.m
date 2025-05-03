function data = load_all_segments(data_folder, subject, segment_type)
    % Load and concatenate all EEG segments of a given type for a subject
    % Input:
    %   data_folder: base path to the dataset
    %   subject: e.g. 'Dog_1'
    %   segment_type: 'ictal', 'interictal', or 'test'
    % Output:
    %   data: concatenated matrix [channels x total_time]

    if nargin < 3
        error('Usage: load_all_segments(data_folder, subject, segment_type)');
    end

    subject_folder = fullfile(data_folder, subject);
    pattern = sprintf('%s_%s_segment_*.mat', subject, segment_type);
    files = dir(fullfile(subject_folder, pattern));

    % Sort by index in filename
    extract_index = @(f) sscanf(f.name, [subject '_' segment_type '_segment_%d.mat']);
    indices = arrayfun(extract_index, files);
    [~, sort_idx] = sort(indices);
    files = files(sort_idx);

    data = [];
    for k = 1:length(files)
        fname = fullfile(subject_folder, files(k).name);
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
