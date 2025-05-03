% Load EEG segment
file = 'H:\Data\PythonDNU\EEG\DataKaggle\Dog_1\Dog_1_ictal_segment_1.mat';
raw = load(file);

% Detect data layout
if isfield(raw, 'data')
    eeg_data = double(raw.data);  % Channels x Time
else
    f = fieldnames(raw);
    eeg_data = double(raw.(f{1}).data);
end

% Plot EEG heatmap
figure;
imagesc(eeg_data);  % Image of raw amplitude
colormap(jet);      % Color scheme
colorbar;

title('Dog 1 - Ictal Segment 1 EEG Heatmap');
xlabel('Time (samples)');
ylabel('Channel');
set(gca, 'YDir', 'normal');  % So channel 1 is on top
