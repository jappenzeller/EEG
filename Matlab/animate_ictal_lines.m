function animateAllIctalPatient1(subjectDir, updatesPerSecond)
% animateAllIctalPatient1  Animate Patient_1’s continuous ictal data.
%
%   animateAllIctalPatient1()  
%       uses default Patient_1 folder and 10 updates/sec.
%   animateAllIctalPatient1(subjectDir)  
%       uses your folder and 10 updates/sec.
%   animateAllIctalPatient1(subjectDir, updatesPerSecond)  
%       lets you choose how many frames per second to draw.
%
% Each frame draws a progressively longer slice of the data,
% simulating real-time playback.

    if nargin<1 || isempty(subjectDir)
        subjectDir = fullfile('H:','Data','PythonDNU','EEG','DataKaggle', ...
                              'Patient_1');
    end
    if nargin<2 || isempty(updatesPerSecond)
        updatesPerSecond = 10;  % draw 10 frames per second
    end

    % --- Load & concatenate all segments ---
    data = loadPatient1Ictal(subjectDir);   % nCh x totalSamples

    % figure out fs from the first segment file
    files = dir(fullfile(subjectDir,'Patient_1_ictal_segment_*.mat'));
    tmp   = load(fullfile(subjectDir, files(1).name), 'data');
    fs    = size(tmp.data,2);               % samples per second

    [nCh, totalSamples] = size(data);
    t = (0:(totalSamples-1)) / fs;

    % vertical offset so channels don’t overlap
    ampMax = max(abs(data(:)));
    offset = ampMax * 1.5;

    % prepare figure & line handles
    figure; hold on;
    lines = gobjects(nCh,1);
    for ch = 1:nCh
        lines(ch) = plot(nan, nan, 'LineWidth', 1);
    end
    xlabel('Time (s)'); ylabel('Channel + offset');
    title(sprintf('Patient\\_1 Ictal Animation (%d channels, fs=%d Hz)', nCh, fs));
    ylim([ -offset, (nCh-1)*offset + offset ]);
    xlim([0, totalSamples/fs]);
    grid on;

    % determine chunk size for each update
    chunkSize = max(1, floor(fs/updatesPerSecond));
    pauseTime = chunkSize / fs;

    % animate
    for idx = chunkSize : chunkSize : totalSamples
        % update each channel’s data
        for ch = 1:nCh
            set(lines(ch), ...
                'XData', t(1:idx), ...
                'YData', data(ch,1:idx) + (ch-1)*offset);
        end
        drawnow;
        pause(pauseTime);
    end

    % draw final remainder if any
    if mod(totalSamples, chunkSize) ~= 0
        for ch = 1:nCh
            set(lines(ch), ...
                'XData', t, ...
                'YData', data(ch,:) + (ch-1)*offset);
        end
        drawnow;
    end
end
