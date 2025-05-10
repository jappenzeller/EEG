% PipelineTest.m
% Uses SeizureDetectionPipelines class to build datasets and perform grid search
% Requires SeizureDetectionPipelines.m in MATLAB path

%% Configuration
subject      = 'Patient_1';
dataRoot     = 'H:/Data/PythonDNU/EEG/DataKaggle';
windowSize   = 400;
step         = 400;

% Hyperparameter grid
treeCounts   = [1000, 2000, 3000];
minLeafSizes = [2, 3, 4, 5];

%% Build feature/label matrix using SeizureDetectionPipelines
fprintf('Building feature matrix for %s...\n', subject);
[X_full, y_full] = SeizureDetectionPipelines.buildFeatureLabelMatrix( ...
    dataRoot, subject, windowSize, step);

%% Cross-validation setup
cv = cvpartition(y_full, 'KFold', 5);
scores = zeros(numel(minLeafSizes), numel(treeCounts));

% Initialize a parallel pool with one worker per CV fold
nFolds = cv.NumTestSets;
if isempty(gcp('nocreate'))
    parpool('local', nFolds);
end

%% Grid search with parfor over folds and AUC calculation
for i = 1:numel(minLeafSizes)
    leafSize = minLeafSizes(i);
    for j = 1:numel(treeCounts)
        nTrees = treeCounts(j);
        fprintf('Evaluating: nTrees=%d, MinLeafSize=%d...\n', nTrees, leafSize);
        foldAUC = zeros(nFolds, 1);
        parfor k = 1:nFolds
            % split indices
            trIdx = training(cv, k);
            teIdx = test(cv, k);

            % Train Random Forest (TreeBagger) without internal parallel
            RF = TreeBagger(nTrees, X_full(trIdx, :), y_full(trIdx), ...
                            'Method',            'classification', ...
                            'MinLeafSize',       leafSize, ...
                            'NumPredictorsToSample','all', ...
                            'OOBPrediction',     'off');

            % Predict and compute AUC
            [~, scoreMat] = predict(RF, X_full(teIdx, :));
            posScores     = scoreMat(:, 2);
            [~, ~, ~, auc] = perfcurve(y_full(teIdx), posScores, 1);
            foldAUC(k)    = auc;
        end
        scores(i, j) = mean(foldAUC);
    end
end

%% Plot heatmap of AUC results
figure;
imagesc(treeCounts, minLeafSizes, scores);
title('Grid Search CV Mean AUC (Patient_1)');
xlabel('Number of Trees');
ylabel('Min Leaf Size');
set(gca, 'YDir', 'normal');
colormap(parula);
colorbar;

% Annotate with values
for i = 1:numel(minLeafSizes)
    for j = 1:numel(treeCounts)
        text(treeCounts(j), minLeafSizes(i), sprintf('%.3f', scores(i, j)), ...
             'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 8);
    end
end
