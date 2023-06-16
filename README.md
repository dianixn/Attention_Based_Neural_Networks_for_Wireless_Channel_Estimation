# Attention_Based_Neural_Networks_for_Wireless_Channel_Estimation
Code for the paper Attention Based Neural Networks for Wireless Channel Estimation HA02

Journal version is provided in https://github.com/dianixn/Channelformer

Cite as - D. Luan and J. Thompson, "Attention Based Neural Networks for Wireless Channel Estimation," 2022 IEEE 95th Vehicular Technology Conference: (VTC2022-Spring), 2022, pp. 1-5, doi: 10.1109/VTC2022-Spring54318.2022.9860803.

%%

"crucial" for "critical". "critical" has a meaning of "vital" from dictionary. Just clarify. 

%%

Attention mechanism, channel estimation, Encoder-decoder archeitecture, deep learning

Propose an encoder-decoder structure (called HA02) which exploits the attention mechanism to focus on the most important input information. In particular, we implement a transformer encoder block as the encoder to achieve the sparsity in the input features and a residual neural network as the decoder respectively. Using 3GPP channel models, our simulations show superior estimation performance compared with other candidate neural network methods for channel estimation.

Run Demonstration_of_H_Rayleigh_Propogation_Channel and Demonstration_of_Pruning_Propogation_Channel for test. 

Run Training.ResNN_pilot_regression to train the ReEsNet and InterpolateNet. 

Run Training.Training_hybrid_simplify to train HA02. 

Run Training.Training_Transformer to train TR structure. 

%% File +parameter has 

		parameters contains the system parameters for generating the training data and testing on the extended SNR

		parameters_doppler contains the system parameters for testing on the extended Doppler shift

		parameters_residual_neural_network contains the hyperparameters for the decoder

%% File +Channel contains 

		Propagation_Channel_Model is a LTEfading channel developed by MATLAB specificed in https://uk.mathworks.com/help/lte/ref/ltefadingchannel.html. 
		We use generalized method of exact Doppler spread method for channel modelling. 

%% File +CSI has

		LS - It is the implementation of the LS method and the time interpolation method is bilinear method. 
		MMSE - It is the linear MMSE method and the time interpolation method is bilinear method. 

%% File +Data_Generation contains

		Data_Generation - used to generate the training data for HA02. 
		Data_Generation_Residual - used to generate the training data for InterpolateNet and ReEsNet
		Data_Generation_Transformer - used to generate the training data for TR method. 

%% File +OFDM contains 

		OFDM_Receiver - OFDM receiver
		OFDM_Transmitter - OFDM transmitter
		Pilot_extract - extract the pilot 
		Pilot_Insert - insert the pilot 
		QPSK_Modualtor - generate QPSK symbols 

%% File +Pruning has

		Encoder_Pruning - used to prune the TR method, but did not show in that paper
		Hybrid_Pruning - used to prune the HA02
		Residual_NN_Pruning - used to prune DAG network (trained InterpolateNet and ReEsNet)
		We did not retrain the neural network after pruning in this paper. 

%% File Residual_NN contains 

		Interpolation_ResNet - Untrained InterpoalteNet (WSA paper)
		Residual_transposed - Untrained ReEsNet

%% File +transformer contains

		model_transformer - system model for TR
		model - system model for HA02
			Encoder_block - the encoder of HA02 
			Decoder_block - the decoder of HA02
		+layer contains the layer modules for attanetion mechanism and residual convolutional neural network
			normalization - layer normalization
			FC1 - fully-connected layer
			gelu - Activation function of GeLu
			multiheadAttention - multihead attention module, which calcualte the attention from Q, K and V
			attention - main control unite of the multiohead attention module, designed by tranformer encoder
			FeedforwardNN - feedforward neural network designed by tranformer encoder

I also attached the trained neural networks, which are Interpolation_ResNet for InterpolateNet, parameters_100 for HA02, parameters_Transformer_1D for TR and ReEsNet for ReEsNet. 

I also attached the pruned neural networks, which are parameters_100_10, parameters_100_10, parameters_100_20, parameters_100_30, parameters_100_40, parameters_100_50,parameters_100_70

%%% Comments 

Run with MATLAB 2021B, with fully-installed deep learning toolbox because it requires customized training. 
