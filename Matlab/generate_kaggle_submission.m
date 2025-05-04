function generate_kaggle_submission(data_folder, subject, fs, win_sec, step_sec, model_seizure, model_early, output_csv)
    % Generate Kaggle submission using seizure and early detection models
    % Inputs:
    %   model_seizure: trained classifier for seizure prediction
    %   model_early: trained classifier for early seizure prediction

    subject_folder = fullfile(data_folder, subject);
    test_files = dir(fullfile(subject_folder, sprintf('%s_test_segment_*.mat', subject)));

    clip_names = strings(length(test_files), 1);
    seizure_probs = zeros(length(test_files), 1);
    early_probs = zeros(length(test_files), 1);

    for k = 1:length(test_files)
        fname = fullfile(subject_folder, test_files(k).name);
        raw = load(fname);
        if isfield(raw, 'data')
            data = double(raw.data);
        else
            f = fieldnames(raw);
            data = double(raw.(f{1}).data);
        end

        % Extract features from entire clip
        windows = segment_into_windows(data, fs, win_sec, step_sec);
        features = extract_features_from_windows(windows, fs);

        % Average predictions across windows
        seizure_pred = mean(predict(model_seizure, features));
        early_pred = mean(predict(model_early, features));

        clip_names(k) = test_files(k).name;
        seizure_probs(k) = seizure_pred;
        early_probs(k) = early_pred;
    end

    T = table(clip_names, seizure_probs, early_probs, 'VariableNames', {'clip', 'seizure', 'early'});
    writetable(T, output_csv);
    fprintf('✅ Kaggle submission written to %s\n', output_csv);
end