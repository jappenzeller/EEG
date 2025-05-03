function model = train_simple_classifier(X, y)
    % Train a simple classifier on EEG features
    % Input:
    %   X: feature matrix [num_samples x num_features]
    %   y: label vector [num_samples x 1]
    % Output:
    %   model: trained classification model

    % Split into training and test sets (80/20)
    cv = cvpartition(y, 'HoldOut', 0.2);
    X_train = X(training(cv), :);
    y_train = y(training(cv));
    X_test = X(test(cv), :);
    y_test = y(test(cv));

    % Train a support vector machine classifier
    model = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', 'Standardize', true);

    % Evaluate on test set
    y_pred = predict(model, X_test);
    accuracy = mean(y_pred == y_test);
    fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

    % Optional: confusion matrix
    figure;
    confusionchart(y_test, y_pred);
    title('Confusion Matrix');
end
