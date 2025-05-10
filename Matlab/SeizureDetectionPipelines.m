% SeizureDetectionPipelines.m
% --------------------------
% MATLAB port of the Python SeizureDetectionPipelines feature‐extraction core.

classdef SeizureDetectionPipelines
    methods (Static)
        function [X, y] = buildFeatureLabelMatrix(rootDir, subject, windowSize, step)
            % Build full-window dataset (label 1=ictal,0=interictal)
            if nargin<3, windowSize = 400; end
            if nargin<4, step       = 400; end
            subjPath = fullfile(rootDir, subject);
            [ictal, interictal] = SeizureDetectionPipelines.loadSubjectSequences(subjPath);
            
            X = [];
            y = [];
            % ictal
            for s = 1:step:(size(ictal,2)-windowSize+1)
                feats = SeizureDetectionPipelines.extractFeaturesMatrix(ictal(:,s:s+windowSize-1));
                X = [X; feats'];
                y = [y; 1];
            end
            % interictal
            for s = 1:step:(size(interictal,2)-windowSize+1)
                feats = SeizureDetectionPipelines.extractFeaturesMatrix(interictal(:,s:s+windowSize-1));
                X = [X; feats'];
                y = [y; 0];
            end
        end
        
        function [X, y] = buildEarlyFeatureLabelMatrix(rootDir, subject, earlySeconds, fs, windowSize)
            % Build early-window dataset (first earlySeconds seconds of ictal)
            if nargin<4, fs         = 400; end
            if nargin<5, windowSize = 400; end
            subjPath = fullfile(rootDir, subject);
            [ictal, interictal] = SeizureDetectionPipelines.loadSubjectSequences(subjPath);
            
            maxSample = min(size(ictal,2), earlySeconds*fs);
            X = [];
            y = [];
            for s = 1:windowSize:(maxSample-windowSize+1)
                feats = SeizureDetectionPipelines.extractFeaturesMatrix(ictal(:,s:s+windowSize-1));
                X = [X; feats'];
                y = [y; 1];
            end
            for s = 1:windowSize:(size(interictal,2)-windowSize+1)
                feats = SeizureDetectionPipelines.extractFeaturesMatrix(interictal(:,s:s+windowSize-1));
                X = [X; feats'];
                y = [y; 0];
            end
        end
        
        function [ictal, interictal] = loadSubjectSequences(subjPath)
            % Load and concatenate all *_ictal_*.mat and *_interictal_*.mat
            ictalFiles    = dir(fullfile(subjPath,'*ictal*.mat'));
            interFiles    = dir(fullfile(subjPath,'*interictal*.mat'));
            ictal    = SeizureDetectionPipelines.concatSorted(ictalFiles,    subjPath);
            interictal = SeizureDetectionPipelines.concatSorted(interFiles, subjPath);
        end
        
        function data = concatSorted(fileList, pathRoot)
            % Extract filenames
            names = {fileList.name};
            % Pull out the number before ".mat" in each filename, e.g. "_12.mat" → 12
            tokens = regexp(names, '_(\d+)\.mat$', 'tokens', 'once');
            % Convert to numeric (you’ll get NaN for any non-matching names)
            nums = cellfun(@(t) str2double(t{1}), tokens);
            % Sort by that number
            [~, order] = sort(nums);
            data = [];
            for k = order
                M = load(fullfile(pathRoot, fileList(k).name));
                fld = fieldnames(M); arr = M.(fld{1});
                if isstruct(arr) && isfield(arr,'data'), arr = arr.data; end
                arr = squeeze(arr);
                data = [data, arr];
            end
        end

        
        function feats = extractFeaturesMatrix(data)
            % Combine FFT, freq-corr and time-corr features into one vector
            fftOut   = SeizureDetectionPipelines.fftFeatures(data);
            freqCorr = SeizureDetectionPipelines.freqCorrFeats(fftOut);
            timeCorr = SeizureDetectionPipelines.timeCorrFeats(data, size(data,2));
            feats    = [fftOut(:); freqCorr; timeCorr];
        end
        
        function out = fftFeatures(data)
            % Log10 magnitude of FFT bins 1–47 Hz
            Y    = fft(data,[],2);
            mags = abs(Y(:,2:48));
            out  = log10(mags + eps);
        end
        
        function feats = freqCorrFeats(fftData)
            % Correlation coefficients + eigenvalues over channels
            Z    = zscore(fftData,0,1);
            C    = corrcoef(Z);
            iu   = triu(true(size(C)),1);
            coeffs = C(iu);
            eigs   = abs(eig(C));
            feats  = [coeffs; sort(eigs)];
        end
        
        function feats = timeCorrFeats(data, targetLen)
            % Resample to targetLen, then corr+eigs
            nCh   = size(data,1);
            origL = size(data,2);
            newD  = zeros(nCh,targetLen);
            xOrig = 1:origL;
            xNew  = linspace(1,origL,targetLen);
            for c=1:nCh
                newD(c,:) = interp1(xOrig,data(c,:),xNew,'linear');
            end
            Z    = zscore(newD,0,1);
            C    = corrcoef(Z);
            iu   = triu(true(size(C)),1);
            coeffs = C(iu);
            eigs   = abs(eig(C));
            feats  = [coeffs; sort(eigs)];
        end
    end
end
