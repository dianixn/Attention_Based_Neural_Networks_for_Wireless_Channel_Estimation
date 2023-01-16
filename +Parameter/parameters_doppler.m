% Parameters

M = 4; % QPSK
k = log2(M);

Num_of_subcarriers = 71; %126
Num_of_FFT = Num_of_subcarriers + 1;
length_of_CP = 16;

Num_of_symbols = 12;
Num_of_pilot = 2;
Frame_size = Num_of_symbols + Num_of_pilot;

Pilot_location_symbols = [1, 13];
Pilot_location = [(1 : 2 : Num_of_FFT)', (2 : 2 : Num_of_FFT)'];
Pilot_value_user = 1 + 1j;

length_of_symbol = Num_of_FFT + length_of_CP;

SNR = 10;

Frequency_Spacing = 15e3;

Carrier_Frequency = 2.1e9;
Max_Mobile_Speed = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; % km/h

SampleRate = Num_of_subcarriers * Frequency_Spacing;
%PathDelays = [0 30 70 90 110 190 410] * 1e-9; % EPA
%AveragePathGains = [0 -1 -2 -3 -8 -17.2 -20.8]; % EPA
%PathDelays = [0 30 150 310 370 710 1090 1730 2510] * 1e-9; % EVA
%AveragePathGains = [0 -1.5 -1.4 -3.6 -0.6 -9.1 -7 -12 -16.9]; % EVA
PathDelays = [0 50 120 200 230 500 1600 2300 5000] * 1e-9; % ETU
AveragePathGains = [-1.0 -1.0 -1.0 0.0 0.0 0.0 -3.0 -5.0 -7.0]; % ETU

DopplerShift = floor((Carrier_Frequency * Max_Mobile_Speed) / (3e8 * 3.6));

DelayProfile = 'ETU'; % 'EPA' 'EVA' 'ETU'
