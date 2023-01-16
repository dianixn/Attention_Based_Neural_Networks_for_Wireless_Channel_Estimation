% Model for TR method

function Z = model_transformer(x, parameters)

w = parameters.Weights;
hyperparameters = parameters.Hyperparameters;

%Z = permute(x, [1 2 4 3]);

Weights = w.decoder_layer.ln_de_w1;
Bias = w.decoder_layer.ln_de_b1;
Z = dlconv(x, Weights, Bias, 'Padding', 'same', 'Stride', [1, 1], 'DataFormat','SSCB');

Z = permute(Z, [1 2 4 3]);

Z = relu(Z);

for i = 1 : hyperparameters.Encoder_num_layers
    Z = transformer.Encoder_block(Z, w.encoder_layer.("layer_"+i), hyperparameters);
end

%Z = transformer.layer.FC1(Z, w.decoder_layer.ln_de_w1, w.decoder_layer.ln_de_b1);

end
