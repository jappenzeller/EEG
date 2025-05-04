function data = load_segment(folder, label, idx)
    fname = fullfile(folder, sprintf('Dog_1_%s_segment_%d.mat', label, idx));
    raw = load(fname);
    if isfield(raw, 'data')
        data = double(raw.data);
    else
        f = fieldnames(raw);
        data = double(raw.(f{1}).data);
    end
end
