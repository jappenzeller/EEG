function combined = loadPatient1Ictal(subjectDir)
% loadPatient1Ictal  Load and concatenate all Patient_1 ictal segments
%
%   combined = loadPatient1Ictal(subjectDir) looks for files named
%   'Patient_1_ictal_segment_#.mat' in subjectDir, sorts by #, and
%   returns a [nChannels × totalSamples] matrix with all segments
%   laid end-to-end.
%
% Example:
%   dataRoot = 'H:\Data\PythonDNU\EEG\DataKaggle\Patient_1';
%   allIctal = loadPatient1Ictal(dataRoot);
%
% Note: each segment is assumed to be 1 s long; combined will be
%       nChannels × (#segments).

    % get list of ictal segment files
    files = dir(fullfile(subjectDir,'Patient_1_ictal_segment_*.mat'));
    if isempty(files)
        error('No Patient_1_ictal_segment_*.mat files found in %s.', subjectDir);
    end

    % extract numeric segment indices
    nums = zeros(numel(files),1);
    for i = 1:numel(files)
        tok = regexp(files(i).name, '_segment_(\d+)\.mat$', 'tokens');
        nums(i) = str2double(tok{1});
    end

    % sort files by their segment number
    [~, order] = sort(nums);
    files = files(order);

    % precompute total number of samples
    % load the first file to get nChannels and segLength
    tmp = load(fullfile(subjectDir, files(1).name), 'data');
    [nCh, segLen] = size(tmp.data);
    totalSamples = segLen * numel(files);

    % preallocate and fill
    combined = zeros(nCh, totalSamples);
    ptr = 1;
    for i = 1:numel(files)
        S = load(fullfile(subjectDir, files(i).name), 'data');
        seg = S.data;   % [nCh × segLen]
        combined(:, ptr:(ptr+segLen-1)) = seg;
        ptr = ptr + segLen;
    end
end
