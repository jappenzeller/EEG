function [ictal_data, interictal_data] = load_all_subject_data(data_folder, subject_label)
    % Load and concatenate all ictal and interictal EEG segments for a subject
    % Example usage: [ictal, interictal] = load_all_subject_data('path/to/data', 'Dog_1');

    if nargin < 2
        error('Usage: load_all_subject_data(data_folder, subject_label)');
    end

    % Get file lists
    ictal_files = dir(fullfile(data_folder, sprintf('%s_ictal_segment_*.mat', subject_label)));
    interictal_files = dir(fullfile(data_folder, sprintf('%s_interictal_segment_*.mat', subject_label)));

    % Sort by segment index
    get_index = @(name, prefix) sscanf(name, [prefix '_%*[^_]_segment_%d.mat']);
    ictal_indices = arrayfun(@(f) get_index(f.name, subject_label), ictal_files);
    interictal_indices = arrayfun(@(f) get_index(f.name, subject_label), interictal_files);
    [~, i_sort] = sort(ictal_indices);
    [~, j_sort] = sort(interictal_indices);
    ictal_files = ictal_files(i_sort);
    interictal_files = interictal_files(j_sort);

    % Load and concatenate
    ictal_data = [];
    interictal_data = [];

    for k = 1:length(ictal_files)
        file = fullfile(data_folder, ictal_files(k).name);
        d = load_segment(file);
        ictal_data = [ictal_data, d];
    end

    for k = 1:length(interictal_files)
        file = fullfile(data_folder, interictal_files(k).name);
        d = load_segment(file);
        interictal_data = [interictal_data, d];
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
