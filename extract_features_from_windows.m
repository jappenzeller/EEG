function feature_matrix = extract_features_from_windows(windows, fs)
    % Extract features for each window using extract_eeg_features
    % Input:
    %   windows: 3D array [channels x win_samples x num_windows]
    %   fs: sampling frequency
    % Output:
    %   feature_matrix: [num_windows x feature_length]

    num_windows = size(windows, 3);
    feature_vec = extract_eeg_features(windows(:, :, 1), fs);
    num_features = length(feature_vec);

    feature_matrix = zeros(num_windows, num_features);
    for i = 1:num_windows
        feature_matrix(i, :) = extract_eeg_features(windows(:, :, i), fs);
    end
end
