function labels = generate_labels(segment_type, num_windows)
    % Generate labels for a given number of windows based on segment type
    % Input:
    %   segment_type: 'ictal', 'interictal', or 'test'
    %   num_windows: number of windows/features extracted
    % Output:
    %   labels: column vector [num_windows x 1]

    switch lower(segment_type)
        case 'ictal'
            labels = ones(num_windows, 1);
        case 'interictal'
            labels = zeros(num_windows, 1);
        case 'test'
            labels = nan(num_windows, 1); % unknown for test
        otherwise
            error('Unknown segment_type: %s', segment_type);
    end
end
