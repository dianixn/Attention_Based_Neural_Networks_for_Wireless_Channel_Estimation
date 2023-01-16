% Residual NN Pruning

Threshold_Value = 0.0591335;

Weight = [];
Weight_Position = [];

for i = 1 : size(DNN_Trained.Layers, 1)

    if contains(DNN_Trained.Layers(i).Name, "conv") || contains(DNN_Trained.Layers(i).Name, "fc")
        Weight_Position = [Weight_Position, i];
    end
    
end

for j = Weight_Position
    
    Weight_of_Layers = reshape(DNN_Trained.Layers(j).Weights, [] ,1);
    Weight = [Weight; Weight_of_Layers];
    
end

Weight_value = sort(abs(Weight));

semilogx(Weight_value, (0 : size(Weight_value, 1) - 1) / (size(Weight_value, 1) - 1));

xlabel('Weight Value');
ylabel('Probability');

Insiginificant_Weight = Weight(abs(Weight) <= Threshold_Value);

Pruning_Location = cell(size(Insiginificant_Weight, 1), 1);

for t = 1 : size(Insiginificant_Weight, 1)
    for k = Weight_Position
        for channel = 1 : size(DNN_Trained.Layers(k).Weights, 3)
            for filter = 1 : size(DNN_Trained.Layers(k).Weights, 4)
                if find(DNN_Trained.Layers(k).Weights(:, :, channel, filter) == Insiginificant_Weight(t))
                    [row, col] = find(DNN_Trained.Layers(k).Weights(:, :, channel, filter) == Insiginificant_Weight(t));
                    Pruning_Location{t, 1} = [k, row, col, channel, filter];
                end
            end
        end
    end
end

Transfer_NN = DNN_Trained.saveobj;

for i = 1 : size(Pruning_Location, 1)
    Transfer_NN.Layers(Pruning_Location{i}(1)).Weights(Pruning_Location{i}(2), Pruning_Location{i}(3), Pruning_Location{i}(4), Pruning_Location{i}(5)) = 0;
end

% Transfer to DAG Net
Pruned_NN = DNN_Trained.loadobj(Transfer_NN);
