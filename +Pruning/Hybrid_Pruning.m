% Encoder and Decoder Pruning

load('parameters_100.mat');

Encoder_pruning = true;
Decoder_pruning = true;

Threshold_Encoder_Ratio = 0.5;
Threshold_Decoder_Ratio = 0.5;

Weight_Encoder = [];
Weight_Decoder = [];

if Encoder_pruning == true
    
    for i = 1 : parameters.Hyperparameters.Encoder_num_layers
        
        Layer_Name = fieldnames(parameters.Weights.encoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)

            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w_")
                Weight_Encoder = [Weight_Encoder; reshape(parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1}), [], 1)];
            end
    
        end
        
    end
    
    Weight_value_Encoder = sort(abs(extractdata(Weight_Encoder)));

    semilogx(Weight_value_Encoder, (0 : size(Weight_value_Encoder, 1) - 1) / (size(Weight_value_Encoder, 1) - 1));

    xlabel('Weight Value');
    ylabel('Probability');

    Threshold_Value_Transformer = Weight_value_Encoder(fix(Threshold_Encoder_Ratio * size(Weight_value_Encoder, 1)));
    
    for i = 1 : parameters.Hyperparameters.Encoder_num_layers
        
        Layer_Name = fieldnames(parameters.Weights.encoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)

            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w_")
                parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1})(abs(parameters.Weights.encoder_layer.("layer_"+i).(Layer_Name{j, 1})) < Threshold_Value_Transformer) = 0;
            end
    
        end
        
    end
    
end

if Decoder_pruning == true
    
    for i = 1 : parameters.Hyperparameters.Decoder_num_layers
        
        Layer_Name = fieldnames(parameters.Weights.decoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)

            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w")
                Weight_Decoder = [Weight_Decoder; reshape(parameters.Weights.decoder_layer.("layer_"+i).(Layer_Name{j, 1}), [], 1)];
            end
    
        end
        
    end
    
    Layer_Name_upsampling = fieldnames(parameters.Weights.decoder_layer.("layer_11"));
        
    for j = 1 : length(Layer_Name_upsampling)

        if contains(Layer_Name_upsampling{j, 1}, "_g_") || contains(Layer_Name_upsampling{j, 1}, "_w")
            Weight_Decoder = [Weight_Decoder; reshape(parameters.Weights.decoder_layer.("layer_11").(Layer_Name_upsampling{j, 1}), [], 1)];
        end
    
    end
    
    Weight_value_Decoder = sort(abs(extractdata(Weight_Decoder)));

    semilogx(Weight_value_Decoder, (0 : size(Weight_value_Decoder, 1) - 1) / (size(Weight_value_Decoder, 1) - 1));

    xlabel('Weight Value');
    ylabel('Probability');

    Threshold_Value_decoder = Weight_value_Decoder(fix(Threshold_Decoder_Ratio * size(Weight_value_Decoder, 1)));
    
    for i = [1 : parameters.Hyperparameters.Decoder_num_layers, 11]
        
        Layer_Name = fieldnames(parameters.Weights.decoder_layer.("layer_"+i));
        
        for j = 1 : length(Layer_Name)
            
            if contains(Layer_Name{j, 1}, "_g_") || contains(Layer_Name{j, 1}, "_w")
                parameters.Weights.decoder_layer.("layer_"+i).(Layer_Name{j, 1})(abs(parameters.Weights.decoder_layer.("layer_"+i).(Layer_Name{j, 1})) < Threshold_Value_decoder) = 0;
            end
    
        end
        
    end
    
end

%save('parameters_100_50', 'parameters')
