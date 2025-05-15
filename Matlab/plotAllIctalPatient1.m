function plotAllIctalPatient1(subjectDir)
% plotAllIctalPatient1  Load and plot all Patient_1 ictal data continuously.
%
%   plotAllIctalPatient1() 
%       uses the default Patient_1 folder under your Kaggle data root.
%
%   plotAllIctalPatient1(subjectDir)
%       loads from the specified folder instead.
%
% This function depends on loadPatient1Ictal(subjectDir) to
% assemble all 1 s segments into one long [nChannels × totalSamples] matrix,
% then plots each channel offset in time.

    % Default folder if none given
    if nargin < 1 || isempty(subjectDir)
        subjectDir = fullfile('H:','Data','PythonDNU','EEG','DataKaggle', ...
                              'Patient_1');
    end

    % Load and concatenate all ictal segments
    data = loadPatient1Ictal(subjectDir);    % [nCh × totalSamples]

    % Infer sampling rate from one segment
    files = dir(fullfile(subjectDir,'Patient_1_ictal_segment_*.mat'));
    if isempty(files)
        error('No Patient_1_ictal_segment_*.mat files found in %s.', subjectDir);
    end
    tmp = load(fullfile(subjectDir,files(1).name),'data');
    fs  = size(tmp.data, 2);  % samples per 1 s → sampling rate in Hz

    % Time vector
    [nCh, nSamp] = size(data);
    t = (0:(nSamp-1)) / fs;

    % Compute vertical offset so traces don’t overlap
    ampMax = max(abs(data(:)));
    offset = ampMax * 1.5;

    % Plot all channels
    figure;
    hold on;
    for ch = 1:nCh
        plot(t, data(ch,:) + (ch-1)*offset, 'LineWidth', 1);
    end
    hold off;

    % Labels and aesthetics
    xlabel('Time (s)');
    ylabel('Channel + offset');
    title(sprintf('Patient_1 Ictal Continuous Data (%d channels, fs=%d Hz)', ...
          nCh, fs), 'Interpreter','none');
    yticks((0:(nCh-1))*offset);
    yticklabels(arrayfun(@num2str, 1:nCh, 'UniformOutput',false));
    grid on;
    xlim([t(1), t(end)]);
end
