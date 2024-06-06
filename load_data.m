function data = load_data(train_source, train_target)
    file_info = dir(train_source);
    data = struct('source', {}, 'target', {});
    cnt = 1;
    for i = 1:length(file_info)
        filename = file_info(i).name;
        if not(endsWith(filename, '.wav'))
            continue
        end
        [data(cnt).source, data(cnt).fs] = audioread(train_source + "/" + filename);
        data(cnt).target = audioread(train_target + "/" + filename);
        cnt = cnt + 1;
    end
end
