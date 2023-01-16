figure;
semilogy(DopplerShift,MSE_LS_over_Doppler, 'Marker', '*', 'LineWidth', 1, 'color', [0 0.4470 0.7410]);
hold on
semilogy(DopplerShift,MSE_MMSE_over_Doppler, 'Marker', 'o', 'LineWidth', 1, 'color', [0.8500 0.3250 0.0980]);
hold on
semilogy(DopplerShift,MSE_DNN_over_Doppler, 'Marker', '+', 'LineWidth', 1, 'LineStyle', ':', 'color', [0.4660 0.6740 0.1880]);
hold on
semilogy(DopplerShift,MSE_ResNet_over_Doppler, 'Marker', 'x', 'LineWidth', 1, 'LineStyle', '--', 'color', [0.6350 0.0780 0.1840]);
hold on
semilogy(DopplerShift,MSE_Transformer_over_Doppler, 'Marker', 's', 'LineWidth', 1, 'color', [0.4940 0.1840 0.5560]);
hold on
semilogy(DopplerShift,MSE_Hybrid_over_Doppler, 'Marker', 'd', 'LineWidth', 1, 'color', [0.3010 0.7450 0.9330]);

ylim([1e-3 1])

legend('LS', ...
    'FD-MMSE', ...
    'Interpolation-ResNet', ...
    'ReEsNet', ...
    'Transformer', ...
    'Hybrid architecture');

xlabel('DopplerShift in Hz');
ylabel('MSE');
title('MSE Performance over the extended Doppler Shift range');
grid on;
hold off;
