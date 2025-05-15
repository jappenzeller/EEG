function plotSpectrogram(matFile, channelIdx)
% plotSpectrogram  Load a 1 s EEG segment and plot its spectrogram.
%
%   plotSpectrogram() uses default file and channel 1.
%   plotSpectrogram(matFile) uses specified file and channel 1.
%   plotSpectrogram(matFile, channelIdx) uses both specified.
%
% Defaults:
%   matFile    = 'H:/Data/PythonDNU/EEG/DataKaggle/Patient_1/Patient_1_ictal_segment_1.mat'
%   channelIdx = 1

    % Handle default arguments
    if nargin < 1 || isempty(matFile)
        matFile = fullfile('H:','Data','PythonDNU','EEG','DataKaggle', ...
                           'Patient_1','Patient_1_ictal_segment_1.mat');
    end
    if nargin < 2 || isempty(channelIdx)
        channelIdx = 1;
    end

    % Load the data
    S = load(matFile);
    if ~isfield(S, 'data')
        error("MAT-file must contain variable 'data' ([nCh × nSamp]).");
    end
    data = S.data;   % [nChannels × nSamples]

    % Infer sampling rate: since each file is exactly 1 s long,
    % fs = number of samples per channel.
    fs = size(data, 2);

    % Validate channel index
    nChannels = size(data,1);
    if channelIdx < 1 || channelIdx > nChannels
        error('channelIdx must be between 1 and %d', nChannels);
    end
    x = data(channelIdx, :);

    % Spectrogram parameters
    windowSec   = 0.5;                  % window length in seconds
    overlapFrac = 0.8;                  % fraction overlap
    window      = round(windowSec * fs);
    noverlap    = round(overlapFrac * window);
    nfft        = max(512, 2^nextpow2(window));

    % Plot spectrogram
    figure;
    spectrogram(x, window, noverlap, nfft, fs, 'yaxis');
    colorbar;
    ylim([0 50]);                       % display up to 50 Hz
    title(sprintf('Spectrogram: %s (Ch %d, fs=%d Hz)', ...
          extractAfter(matFile, filesep), channelIdx, fs));
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    drawnow;
end
