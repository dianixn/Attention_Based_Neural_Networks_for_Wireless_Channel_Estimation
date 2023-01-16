figure;
plot(SNR_Range,Denoising_gain, 'Marker', '*', 'LineWidth', 1, 'color', [0 0.4470 0.7410]);
hold on
plot(SNR_Range,Denoising_gain_10, 'Marker', 'o', 'LineWidth', 1, 'color', [0.8500 0.3250 0.0980]);
hold on
plot(SNR_Range,Denoising_gain_20, 'Marker', '+', 'LineWidth', 1, 'LineStyle', ':', 'color', [0.4660 0.6740 0.1880]);
hold on
plot(SNR_Range,Denoising_gain_30, 'Marker', 'x', 'LineWidth', 1, 'LineStyle', '--', 'color', [0.6350 0.0780 0.1840]);
hold on
plot(SNR_Range,Denoising_gain_50, 'Marker', 's', 'LineWidth', 1, 'color', [0.4940 0.1840 0.5560]);
hold on
plot(SNR_Range,Denoising_gain_70, 'Marker', 'v', 'LineWidth', 1, 'LineStyle', '-.', 'color', [0.9290 0.6940 0.1250]);

ylim([-5 20])

legend('HA02', ...
    'HA02 with 10% pruning', ...
    'HA02 with 20% pruning', ...
    'HA02 with 30% pruning', ...
    'HA02 with 50% pruning', ...
    'HA02 with 70% pruning');
xlabel('SNR in dB');
ylabel('Denoising gain in dB');
title('Denoising gain and weight-level pruning');
grid on;
hold off;
