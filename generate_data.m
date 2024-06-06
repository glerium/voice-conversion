train_data = load_data('./data/train/source', './data/train/target');
for i = 1:length(train_data)
    train_data(i).source = pre_emphasis(train_data(i).source);
    train_data(i).target = pre_emphasis(train_data(i).target);
end

for i = 1:length(train_data)
    train_data(i).source_graph = melSpectrogram(train_data(i).source, train_data(i).fs);
    train_data(i).target_graph = melSpectrogram(train_data(i).target, train_data(i).fs);
end

save('train_data.mat', 'train_data')
