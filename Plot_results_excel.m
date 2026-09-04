% --- SCRIPT MATLAB PER PLOT KPI STRESS TESTS CON ESPORTAZIONE ---
clear all; close all; clc;

%% 1. DATI ESTRATTI DALL'ANALISI 

% Test 1: Frequenza di Generazione (Tempo tra gli spawn in secondi)
freq_val = [0.75, 0.60, 0.50, 0.40, 0.30];
freq_sing = [100, 97.78, 77.78, 65.56, 64.44];
freq_pph = [8842, 10355, 11674, 13377, 15895];
freq_gap = [0.477, 0.361, 0.292, 0.215, 0.114];

% Test 2: Numero Totale di Pacchi (Stress Test di Resistenza)
tot_val = [10, 20, 40, 60, 80, 100];
tot_sing = [100, 100, 100, 99.83, 99.49, 99.90];
tot_pph = [8842, 9245, 9417, 9482, 9520, 9529];
tot_gap = [0.477, 0.493, 0.490, 0.495, 0.480, 0.482];

% Test 3: Velocità Tappeto (Effetto Estrazione in m/s)
vel_val = [1.90, 1.75, 1.50, 1.00, 0.80, 0.60];
vel_sing = [100, 100, 100, 96.67, 73.33, 51.11];
vel_pph = [8842, 8842, 8842, 8842, 8829, 8853];
vel_gap = [0.477, 0.421, 0.327, 0.140, 0.060, -0.004];

% Test 4: Box per Generazione (Ammassamento Iniziale)
box_val = [2, 3, 4, 5, 6];
box_sing = [100, 74.44, 63.33, 50.00, 45.56];
box_pph = [8842, 11074, 14410, 20661, 20582];
box_gap = [0.477, 0.321, 0.171, 0.023, 0.023];

%% 2. IMPOSTAZIONE AUTOMATICA DELLE FIGURE E SALVATAGGIO

% Matrice di configurazione: [x, sing, pph, gap, xlabel, title, invert_x_axis, filename]
params = {
    freq_val, freq_sing, freq_pph, freq_gap, 'Box generation period [s]', 'Density effect', true, 'StressTest_1_DensitaTraffico.png';
    tot_val, tot_sing, tot_pph, tot_gap, 'Total number of parcels', 'Long term stability', false, 'StressTest_2_ResistenzaLungoTermine.png';
    vel_val, vel_sing, vel_pph, vel_gap, 'Treadmill velocity [m/s]', 'Extraction effect', true, 'StressTest_3_EffettoEstrazione.png';
    box_val, box_sing, box_pph, box_gap, 'Boxes per generation', 'Initial bottleneck effect', false, 'StressTest_4_AmmassamentoIniziale.png'
};

disp("Generazione ed esportazione dei grafici in corso...");

for i = 1:4
    x = params{i, 1};
    sing = params{i, 2};
    pph = params{i, 3};
    gap = params{i, 4};
    x_label = params{i, 5};
    fig_title = params{i, 6};
    invert_x = params{i, 7}; 
    filename = params{i, 8};

    % Creazione finestra panoramica proporzionata
    fig = figure('Name', fig_title, 'Color', 'w', 'Position', [100, 100, 1300, 400]);
    
    % --- PLOT 1: Tasso di Singolarizzazione ---
    subplot(1, 3, 1);
    plot(x, sing, '-o', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', 'w', 'MarkerSize', 6);
    grid on;
    xlabel(x_label, 'FontWeight', 'bold');
    ylabel('Singularization rate (%)', 'FontWeight', 'bold');
    title('Success rate');
    ylim([0 105]);
    if invert_x, set(gca, 'XDir', 'reverse'); end % Inverte asse se il caso peggiore è a valori bassi
    
    % --- PLOT 2: Throughput ---
    subplot(1, 3, 2);
    plot(x, pph, '-s', 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980], 'MarkerFaceColor', 'w', 'MarkerSize', 6);
    grid on;
    xlabel(x_label, 'FontWeight', 'bold');
    ylabel('Throughput (Parcels/Hour)', 'FontWeight', 'bold');
    title('Flow Efficiency');
    if invert_x, set(gca, 'XDir', 'reverse'); end

    % --- PLOT 3: Average Gap ---
    subplot(1, 3, 3);
    plot(x, gap, '-d', 'LineWidth', 2.5, 'Color', [0.4660 0.6740 0.1880], 'MarkerFaceColor', 'w', 'MarkerSize', 6);
    hold on;
    yline(0.15, 'r--', 'Target (0.15m)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
    yline(0, 'k-', 'Collision limit', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
    grid on;
    xlabel(x_label, 'FontWeight', 'bold');
    ylabel('Average gap [m]', 'FontWeight', 'bold');
    title('Vertical gap');
    if invert_x, set(gca, 'XDir', 'reverse'); end

    % Titolo Globale per la figura
    sgtitle(fig_title, 'FontSize', 15, 'FontWeight', 'bold');
    
    % Forza l'aggiornamento visivo prima del salvataggio
    drawnow;
    
    % ESPORTAZIONE PROFESSIONALE
    % Salva la figura corrente (fig) nel file specificato a 300 DPI
    exportgraphics(fig, filename, 'Resolution', 300, 'BackgroundColor', 'w');
    
    fprintf('Salvato con successo: %s\n', filename);
end

disp("Tutti i grafici sono stati esportati nella cartella corrente di MATLAB.");