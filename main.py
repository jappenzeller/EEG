import argparse
from pathlib import Path
import pandas as pd

from pipeline import train_model, predict_test
from submit import write_submission, kaggle_submit


def main():
    parser = argparse.ArgumentParser(
        description='Train and optionally predict and submit EEG seizure detection results.'
    )
    parser.add_argument(
        '--data-root', default='H:/Data/PythonDNU/EEG/DataKaggle',
        help='Root directory containing subject subfolders'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=['patient_1'],
        help="List of subjects (default: ['patient_1'])"
    )
    parser.add_argument(
        '--predict', action='store_true', default=False,
        help='Whether to run predictions on test segments'
    )
    parser.add_argument(
        '--submit', action='store_true', default=False,
        help='Whether to write submission file and submit to Kaggle'
    )
    parser.add_argument(
        '--comp', default='seizure-detection',
        help='Kaggle competition slug'
    )
    args = parser.parse_args()

    # Grid search parameters
    param_grid = {
        'rf__n_estimators': [3000,4000,5000],
        'rf__min_samples_split': [2,3,4]
    }
    cv_folds = 5

    all_submissions = []

    for subject in args.subjects:
        subject_dir = Path(args.data_root) / subject
        print(f"=== Processing {subject} ===")

        # Determine sampling rate
        if subject.lower().startswith('patient'):
            fs = 500
            window_sec = 1
        elif subject.lower().startswith('dog'):
            fs = 400
            window_sec = 1
        else:
            raise ValueError(f"Unknown subject type: {subject}")

        # Train model
        model, best_params = train_model(subject_dir, fs, param_grid, cv_folds, window_sec)
        print(f"Trained {subject} model (fs={fs}), best params: {best_params}")

        # Predict on test data if requested
        if args.predict:
            df_sub = predict_test(subject_dir, fs, model)
            all_submissions.append(df_sub)

    # Write submission and/or submit if requested
    if args.submit:
        if not args.predict:
            print("--submit requires --predict to generate predictions. Skipping.")
        else:
            submission_df = pd.concat(all_submissions, ignore_index=True)
            write_submission(submission_df, filename='submission.csv')
            kaggle_submit('submission.csv', 'Automated submission', args.comp)


if __name__ == '__main__':
    main()
