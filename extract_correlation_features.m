function features = extract_correlation_features(filepath)
    % Load EEG segment
    mat_data = load(filepath);
    top_field = fieldnames(mat_data);
    seg = mat_data.(top_field{1});
    
    data = double(seg.data);  % Channels x Time
    fs = double(seg.sampling_frequency);
    [n_channels, n_samples] = size(data);

    % --- TIME DOMAIN FEATURES ---
    % Correlation matrix over time (channels x channels)
    R_time = corrcoef(data');
    eig_time = sort(eig(R_time), 'descend');  % Eigenvalues in descending order

    % --- FREQUENCY DOMAIN FEATURES ---
    % Band-limited FFT magnitude
    NFFT = 2^nextpow2(n_samples);
    freq_data = abs(fft(data, NFFT, 2));  % Channels x Frequency
    freq_data = freq_data(:, 1:NFFT/2);   % Keep positive frequencies

    % Normalize power spectrum (per channel)
    freq_data = freq_data ./ max(freq_data, [], 2);

    % Correlation of spectral content across channels
    R_freq = corrcoef(freq_data');
    eig_freq = sort(eig(R_freq), 'descend');

    % Combine features: take top k eigenvalues from each domain
    k = min(5, length(eig_time));  % or 10 for more
    features = [eig_time(1:k), eig_freq(1:k)];
end
