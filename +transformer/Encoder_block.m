function [Z, present] = Encoder_block(X, weights, hyperParameters)

[A, present] = transformer.layer.attention(X, [], weights, hyperParameters); %(numFeatures*numHeads)-by-numInputSubwords-by-1

A = A + X; 

A = transformer.layer.normalization(A, ...
    weights.ln_1_g_0, weights.ln_1_b_0);

Z = transformer.layer.FeedforwardNN(A, weights);

Z = Z + A;

Z = transformer.layer.normalization(Z, ...
    weights.ln_2_g_0, weights.ln_2_b_0);

end