subjects = {
    'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', ...
    'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', ...
    'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8'
};

data_folder = 'H:\Data\PythonDNU\EEG\DataKaggle';
fs = 400;
win_sec = 1.0;
step_sec = 0.5;

all_submissions = table();

for i = 1:length(subjects)
    subject = subjects{i};
    fprintf('📦 Processing %s...\n', subject);

    % Build seizure model
    [X, y] = build_subject_dataset(data_folder, subject, fs, win_sec, step_sec);
    model_seizure = train_simple_classifier(X, y);

    % Build early detection model
    [~, ~, model_early] = build_early_detection_dataset(data_folder, subject, fs, win_sec, step_sec);

    % Generate submission table
    output_csv = sprintf('%s_submission.csv', lower(subject));
    generate_kaggle_submission(data_folder, subject, fs, win_sec, step_sec, model_seizure, model_early, output_csv);

    % Load and append submission rows
    T = readtable(output_csv);
    all_submissions = [all_submissions; T];
end

% Save final combined submission
writetable(all_submissions, 'kaggle_combined_submission.csv');
fprintf('✅ All submissions saved to kaggle_combined_submission.csv\n');
