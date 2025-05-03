function features = extract_eeg_features(data, fs)
    % Extract EEG features based on correlation matrix eigenvalues
    % in both time and frequency domains for one EEG segment
    % Input:
    %   data: matrix [channels x time]
    %   fs: sampling frequency
    % Output:
    %   features: concatenated feature vector from time and frequency domain

    % --- 1. Time Domain ---
    corr_time = corrcoef(data');  % correlation between channels
    eig_time = sort(eig(corr_time), 'descend');

    % --- 2. Frequency Domain ---
    N = size(data, 2);
    fft_data = abs(fft(data, [], 2));
    fft_data = fft_data(:, 1:floor(N/2));  % keep positive frequencies
    corr_freq = corrcoef(fft_data');
    eig_freq = sort(eig(corr_freq), 'descend');

    % Concatenate features
    features = [eig_time; eig_freq];
end
