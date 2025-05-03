function [X_early, y_early, model_early] = build_early_detection_dataset(data_folder, subject, fs, win_sec, step_sec)
    % Build and train model for early seizure detection using early ictal windows
    % Output:
    %   X_early: feature matrix for early detection model
    %   y_early: label vector (1 for early, 0 for later in seizure)
    %   model_early: trained SVM model for early detection

    % Load ictal segments
    ictal_data = load_all_segments(data_folder, subject, 'ictal');
    windows = segment_into_windows(ictal_data, fs, win_sec, step_sec);

    % Determine number of early windows based on 15 sec threshold
    samples_per_win = round(win_sec * fs);
    samples_per_step = round(step_sec * fs);
    early_cutoff = 15 * fs;
    total_samples = size(ictal_data, 2);
    num_windows = size(windows, 3);

    y_early = zeros(num_windows, 1);
    for i = 1:num_windows
        win_start_sample = (i - 1) * samples_per_step + 1;
        if win_start_sample <= early_cutoff
            y_early(i) = 1;
        end
    end

    X_early = extract_features_from_windows(windows, fs);

    % Train early detection model
    rng(1);
    cv = cvpartition(y_early, 'HoldOut', 0.2);
    X_train = X_early(training(cv), :);
    y_train = y_early(training(cv));
    X_test = X_early(test(cv), :);
    y_test = y_early(test(cv));

    model_early = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', 'Standardize', true);

    % Evaluate
    y_pred = predict(model_early, X_test);
    acc = mean(y_pred == y_test);
    fprintf('Early Detection Model Accuracy: %.2f%%\n', acc * 100);
    figure;
    confusionchart(y_test, y_pred);
    title('Early Detection Confusion Matrix');
end
