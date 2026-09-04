% Carica i dati del training
load('trainingStats.mat', 'trainingStats');

episodes = trainingStats.EpisodeIndex;
rewards = trainingStats.EpisodeReward;
avg_rewards = trainingStats.AverageReward;

%% Reward in funzione degli episodes
% Crea una figura professionale con sfondo bianco
fig_reward = figure('Name', 'Training Reward', 'Color', 'w', 'Position', [100, 100, 800, 500]);

% Plotta i dati
plot(episodes, rewards, 'Color', [0.6 0.8 1.0], 'LineWidth', 0.5); hold on; % Reward grezza
plot(episodes, avg_rewards, 'Color', [0 0.4470 0.7410], 'LineWidth', 2.5); % Reward media
grid on;

% Formattazione per paper/presentazione
xlabel('Episode', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Cumulative Reward', 'FontSize', 12, 'FontWeight', 'bold');
title('PPO Training Performance', 'FontSize', 14);
legend('Episode Reward', 'Moving Average', 'Location', 'southeast', 'FontSize', 11);

% Salva come immagine PNG ad alta risoluzione (300 DPI)
exportgraphics(fig_reward, 'Reward_Curve_HighRes.png', 'Resolution', 300);


%% Reward in funzione degli updates
% Estrazione dei dati grezzi
steps_per_episode = trainingStats.EpisodeSteps; % Quanti step fisici per ogni episodio

% 1. Calcolo degli step totali cumulativi
total_steps = cumsum(steps_per_episode);

% 2. Conversione da Step a "Policy Updates" (come nel paper)
% Sapendo che il PPO fa un update ogni 2048 step (ExperienceHorizon)
updates = total_steps / 2048;

% Creazione del grafico stile "Paper"
fig_paper = figure('Name', 'Training Reward (Paper Style)', 'Color', 'w', 'Position', [100, 100, 800, 500]);

% Plot con asse X in funzione degli Updates
plot(updates, rewards, 'Color', [0.6 0.8 1.0], 'LineWidth', 0.5); hold on;
plot(updates, avg_rewards, 'Color', [0 0.4470 0.7410], 'LineWidth', 2.5);
grid on;

% Formattazione
xlabel('Training Updates', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Cumulative Reward', 'FontSize', 12, 'FontWeight', 'bold');
title('PPO Training Performance', 'FontSize', 14);
legend('Episode Reward', 'Moving Average', 'Location', 'southeast', 'FontSize', 11);

% Esporta
exportgraphics(fig_paper, 'Reward_vs_Updates.png', 'Resolution', 300);

%% Plot della Q0 function - prova
q0_values = trainingStats.EpisodeQ0;

fig_loss = figure('Name', 'Episode Q0', 'Color', 'w', 'Position', [150, 150, 800, 500]);
plot(episodes, q0_values, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5);
grid on;
xlabel('Episode', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Episode Q0 (Critic Estimate)', 'FontSize', 12, 'FontWeight', 'bold');
title('Critic Value Estimate Evolution', 'FontSize', 14);

exportgraphics(fig_loss, 'Critic_Q0_Curve.png', 'Resolution', 300);