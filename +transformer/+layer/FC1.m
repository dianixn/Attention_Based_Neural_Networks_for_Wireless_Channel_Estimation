function Z = FC1(X, W, b)

% Fully connected layer

Z = dlmtimes(W,X) + b;

end