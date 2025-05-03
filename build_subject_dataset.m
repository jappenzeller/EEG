function [X, y] = build_subject_dataset(data_folder, subject, fs, win_sec, step_sec)
    % Full pipeline to load ictal + interictal data, extract features, shuffle, and save
    % Input:
    %   data_folder: path to dataset
    %   subject: e.g. 'Dog_1'
    %   fs: sampling frequency
    %   win_sec: window length in seconds
    %   step_sec: step size between windows in seconds
    % Output:
    %   X: feature matrix [num_windows x num_features]
    %   y: label vector [num_windows x 1]

    % Load ictal
    ictal_data = load_all_segments(data_folder, subject, 'ictal');
    ictal_windows = segment_into_windows(ictal_data, fs, win_sec, step_sec);
    X_ictal = extract_features_from_windows(ictal_windows, fs);
    y_ictal = generate_labels('ictal', size(X_ictal, 1));

    % Load interictal
    interictal_data = load_all_segments(data_folder, subject, 'interictal');
    interictal_windows = segment_into_windows(interictal_data, fs, win_sec, step_sec);
    X_interictal = extract_features_from_windows(interictal_windows, fs);
    y_interictal = generate_labels('interictal', size(X_interictal, 1));

    % Combine and shuffle
    X = [X_ictal; X_interictal];
    y = [y_ictal; y_interictal];

    rng(1);  % for reproducibility
    idx = randperm(size(X, 1));
    X = X(idx, :);
    y = y(idx);

    % Save dataset to .mat file
    save_filename = sprintf('%s_combined_dataset.mat', lower(subject));
    save(fullfile(data_folder, save_filename), 'X', 'y');
    fprintf('✅ Saved combined dataset to %s\n', fullfile(data_folder, save_filename));
end