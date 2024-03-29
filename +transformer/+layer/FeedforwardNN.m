function Z = FeedforwardNN(X, weights)

Z = transformer.layer.FC1( X, weights.mlp_c_fc_w_0, weights.mlp_c_fc_b_0 );
Z = transformer.layer.gelu(Z);
Z = transformer.layer.FC1(Z, weights.mlp_c_proj_w_0, weights.mlp_c_proj_b_0 );

end