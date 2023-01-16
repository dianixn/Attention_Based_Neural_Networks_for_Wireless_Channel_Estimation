Number_of_filters = 2;

filterSize = [2 2];
numChannels = 1;
numFilters = Number_of_filters;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w = initializeGlorot(sz, numOut, numIn);
parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_b = dlarray(zeros(numFilters, 1));

for j = 1 : parameters.Hyperparameters.Decoder_num_layers
    
    filterSize = [2 2];
    numChannels = Number_of_filters;
    numFilters = Number_of_filters;

    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;

    parameters.Weights.decoder_layer.("layer_" + j).ln_de_w1 = initializeGlorot(sz, numOut, numIn);
    parameters.Weights.decoder_layer.("layer_" + j).ln_de_b1 = dlarray(zeros(numFilters, 1));
    
    filterSize = [2 2];
    numChannels = Number_of_filters;
    numFilters = Number_of_filters;

    sz = [filterSize numChannels numFilters];
    numOut = prod(filterSize) * numFilters;
    numIn = prod(filterSize) * numFilters;

    parameters.Weights.decoder_layer.("layer_" + j).ln_de_w2 = initializeGlorot(sz, numOut, numIn);
    parameters.Weights.decoder_layer.("layer_" + j).ln_de_b2 = dlarray(zeros(numFilters, 1));
    
    parameters.Weights.decoder_layer.("layer_"+j).ln_de_w3 = initializeGlorot([Feature_size, 1], prod([Feature_size, 1]), prod([Feature_size, 1]));
    parameters.Weights.decoder_layer.("layer_"+j).ln_de_b3 = dlarray(zeros(Feature_size, 1));
    
end

parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w1 = initializeGlorot([size(Training_Y, 1), Feature_size], prod([size(Training_Y, 1), Feature_size]), prod([size(Training_Y, 1), Feature_size]));
parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_b1 = dlarray(zeros(size(Training_Y, 1), 1));
parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w2 = initializeGlorot([size(Training_Y, 1), size(Training_Y, 1)], prod([size(Training_Y, 1), size(Training_Y, 1)]), prod([size(Training_Y, 1), size(Training_Y, 1)]));
parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_b2 = dlarray(zeros(size(Training_Y, 1), 1));

filterSize = [2 2];
numChannels = Number_of_filters;
numFilters = 1;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_w0 = initializeGlorot(sz, numOut, numIn);
parameters.Weights.decoder_layer.("layer_" + parameters.Hyperparameters.Decoder_num_layers + 1).ln_de_b0 = dlarray(zeros(numFilters, 1));

function weights = initializeGlorot(sz, numOut, numIn)

Z = 2 * rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end
