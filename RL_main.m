clear all
close all
clc

%% ================================================================
%  RL MAIN - Parcel Singulation with PPO
%  Author: Matteo
%  Description:
%  Training and testing script for PPO agent controlling a 5x5 AMS
%  parcel singulation system.
%% ================================================================

rng(10)
%% Inizializzazione GPU
if canUseGPU
    gpuDevice(1); % Seleziona la NVIDIA GTX 1650
    disp("NVIDIA GPU rilevata e pronta per l'addestramento.");
else
    disp("Attenzione: GPU non rilevata. Il calcolo avverrà su CPU.");
end
%% Create environment

env = RL_environment();

% Optional but useful check
validateEnvironment(env);

actInfo = getActionInfo(env);
obsInfo = getObservationInfo(env);

numObs = prod(obsInfo.Dimension);
numAct = prod(actInfo.Dimension);

fprintf("Observation dimension: %d\n", numObs);
fprintf("Action dimension: %d\n", numAct);

%% Training mode

doTraining = false;

% Choose:
% "fast"   = quick debug
% "medium" = useful training test
% "final"  = long final training
trainingLevel = "medium";

%% Neural network sizes

criticLayerSizes = [512 256 128];
actorLayerSizes  = [512 256 128];

%% ================================================================
%  CRITIC NETWORK
%% ================================================================

criticNetwork = [
    featureInputLayer(numObs, Name="criticInput")

    fullyConnectedLayer(criticLayerSizes(1), ...
        Weights=sqrt(2/numObs)*(rand(criticLayerSizes(1),numObs)-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(1),1))
    reluLayer

    fullyConnectedLayer(criticLayerSizes(2), ...
        Weights=sqrt(2/criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(2),1))
    reluLayer

    fullyConnectedLayer(criticLayerSizes(3), ...
        Weights=sqrt(2/criticLayerSizes(2))*(rand(criticLayerSizes(3),criticLayerSizes(2))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(3),1))
    reluLayer

    fullyConnectedLayer(1, ...
        Weights=sqrt(2/criticLayerSizes(3))*(rand(1,criticLayerSizes(3))-0.5), ...
        Bias=1e-3)
];

criticNetwork = dlnetwork(criticNetwork);
summary(criticNetwork)

critic = rlValueFunction(criticNetwork, obsInfo);

%% ================================================================
%  ACTOR NETWORK
%% ================================================================

commonPath = [
    featureInputLayer(numObs, Name="netOin")

    fullyConnectedLayer(actorLayerSizes(1), Name="actorFC1")
    reluLayer(Name="actorRelu1")

    fullyConnectedLayer(actorLayerSizes(2), Name="actorFC2")
    reluLayer(Name="relulast")
];

meanPath = [
    fullyConnectedLayer(actorLayerSizes(3), Name="MeanLyr")
    reluLayer(Name="meanRelu")
    fullyConnectedLayer(numAct, Name="meanOutLyr")
    tanhLayer(Name="thmeanOutLyr")
];

stdPath = [
    fullyConnectedLayer(actorLayerSizes(3), Name="StdLyr")
    reluLayer(Name="stdRelu")
    fullyConnectedLayer(numAct, Name="stdFCLyr")
    softplusLayer(Name="stdOutLyr")
];

actorGraph = layerGraph(commonPath);
actorGraph = addLayers(actorGraph, meanPath);
actorGraph = addLayers(actorGraph, stdPath);

actorGraph = connectLayers(actorGraph, "relulast", "MeanLyr/in");
actorGraph = connectLayers(actorGraph, "relulast", "StdLyr/in");

actorNetwork = dlnetwork(actorGraph);
summary(actorNetwork)

actor = rlContinuousGaussianActor(actorNetwork, obsInfo, actInfo, ...
    ActionMeanOutputNames="thmeanOutLyr", ...
    ActionStandardDeviationOutputNames="stdOutLyr", ...
    ObservationInputNames="netOin");

%% ================================================================
%  PPO OPTIONS
%% ================================================================

% actorOpts = rlOptimizerOptions(LearnRate=1e-4, GradientThreshold=1);
% criticOpts = rlOptimizerOptions(LearnRate=1e-4, GradientThreshold=1);
actorOpts = rlOptimizerOptions(LearnRate=5e-5, GradientThreshold=1);
criticOpts = rlOptimizerOptions(LearnRate=5e-5, GradientThreshold=1);
%NUMEPOCH PRIMA ERA 30
agentOpts = rlPPOAgentOptions( ...
    ExperienceHorizon=2048, ...
    ClipFactor=0.2, ...
    EntropyLossWeight=0.0001, ...
    ActorOptimizerOptions=actorOpts, ...
    CriticOptimizerOptions=criticOpts, ...
    NumEpoch=10, ...
    MiniBatchSize=256, ...
    AdvantageEstimateMethod="gae", ...
    GAEFactor=0.95, ...
    SampleTime=0.01, ...
    DiscountFactor=0.99);

agent = rlPPOAgent(actor, critic, agentOpts);

%% ================================================================
%  TRAINING OPTIONS
%% ================================================================

switch trainingLevel

    case "fast"

        trainOpts = rlTrainingOptions( ...
            MaxEpisodes=30, ...
            MaxStepsPerEpisode=300, ...
            Plots="training-progress", ...
            StopTrainingCriteria="EpisodeCount", ...
            StopTrainingValue=30, ...
            ScoreAveragingWindowLength=5, ...
            SaveAgentCriteria="EpisodeReward", ...
            SaveAgentValue=500, ...
            SaveAgentDirectory="savedAgents_fast");

    case "medium"

        trainOpts = rlTrainingOptions( ...
            MaxEpisodes=500, ...
            MaxStepsPerEpisode=1000, ...
            Plots="training-progress", ...
            StopTrainingCriteria="EpisodeCount", ...
            StopTrainingValue=1500, ...
            ScoreAveragingWindowLength=50, ...
            SaveAgentCriteria="EpisodeReward", ...
            SaveAgentValue=1200, ...
            SaveAgentDirectory="savedAgents_medium",...
            UseParallel = true);

    case "final"

        trainOpts = rlTrainingOptions( ...
            MaxEpisodes=25000, ...
            MaxStepsPerEpisode=1000, ...
            Plots="training-progress", ...
            StopTrainingCriteria="AverageReward", ...
            StopTrainingValue=5000, ...
            ScoreAveragingWindowLength=100, ...
            SaveAgentCriteria="EpisodeReward", ...
            SaveAgentValue=1500, ...
            SaveAgentDirectory="savedAgents_final");

    otherwise

        error("trainingLevel must be 'fast', 'medium', or 'final'");
end

%% ================================================================
%  TRAIN OR LOAD
%% ================================================================

if doTraining == true
    
    trainingStats = train(agent, env, trainOpts);

    save("agent_trained.mat", "agent");
    save("trainingStats.mat", "trainingStats");

else

    load("agent_trained_final.mat", "agent");

end

%% ================================================================
%  CUSTOM EVALUATION (KPIs) - NUOVO
%% ================================================================
rng(42); 
disp("Inizio simulazione per calcolo KPI...");
num_test_episodes = 10; 
throughput_list = zeros(num_test_episodes, 1);
singulation_list = zeros(num_test_episodes, 1);
mean_gap_ep_list = zeros(num_test_episodes, 1); 

disp("Avvio registrazione video (Solo Episodio 1)...");
v = VideoWriter('Parcel_Sorting_Animation.mp4', 'MPEG-4');
v.FrameRate = 33; 
v.Quality = 100;
open(v);

for ep = 1:num_test_episodes
    
    obs = reset(env);
    isDone = false;
    step_count = 0;
    frame_skip = 3; 
    
    % Forza l'apertura della finestra prima del loop
    if ep == 1
        plot(env);
    end
    
    while ~isDone
        action = getAction(agent, obs);
        [obs, reward, isDone, info] = step(env, action{1});
        step_count = step_count + 1;
        
        % Registra il video SOLO durante il primo episodio
        if ep == 1 && mod(step_count, frame_skip) == 0
            plot(env); 
            drawnow;   

            % SOLUZIONE: findall(0, ...) cerca in tutta la root ignorando l'HandleVisibility nascosto
            fig = findall(0, 'Type', 'figure', 'Name', 'AMS sorting visualizer simple');

            % Se non trova la versione simple, cerca la versione completa
            if isempty(fig)
                fig = findall(0, 'Type', 'figure', 'Name', 'AMS sorting visualizer');
            end

            if ~isempty(fig)
                % Seleziona la prima figura trovata e cattura il frame
                frame = getframe(fig(1));
                writeVideo(v, frame);
            end
        end
    end
    
    % Chiudi il video appena finisce il primo episodio
    if ep == 1
        close(v);
        disp("Video salvato con successo come 'Parcel_Sorting_Animation.mp4'");
    end
    
    %% 1. Tasso di Singolarizzazione
    singulation_list(ep) = env.computeFinalSingulationScore();
    
    %% 2. Calcolo Gap Medio dell'Episodio
    if env.index_exit >= 2
        gaps = zeros(env.index_exit - 1, 1);
        for q = 1:env.index_exit - 1
            a = env.exit_order(q);
            b = env.exit_order(q+1);
            gap = abs(env.y_boxes(b) - env.y_boxes(a)) - env.d_boxes(a)/2 - env.d_boxes(b)/2;
            gaps(q) = gap;
        end
        mean_gap_ep_list(ep) = mean(gaps);
    else
        mean_gap_ep_list(ep) = 0; 
    end
    
    %% 3. Throughput
    total_time = env.time;
    total_exited_packages = env.index_exit;
    if total_time > 0
        throughput_sec = total_exited_packages / total_time;
        throughput_list(ep) = throughput_sec * 3600; 
    else
        throughput_list(ep) = 0;
    end
    
    fprintf('Episodio %d: Singol. = %.2f%%, Gap Medio = %.3f m, Throughput = %d IPH\n', ...
            ep, singulation_list(ep)*100, mean_gap_ep_list(ep), round(throughput_list(ep)));
end

% Risultati medi finali
mean_singulation = mean(singulation_list);
mean_throughput = mean(throughput_list);
mean_gap_total = mean(mean_gap_ep_list); 
fprintf('\n=== RISULTATI FINALI KPI (Media su %d episodi) ===\n', num_test_episodes);
fprintf('Tasso di Singolarizzazione Medio: %.2f %%\n', mean_singulation * 100);
fprintf('Throughput Medio: %d pacchi/ora\n', round(mean_throughput));
fprintf('Gap Medio Totale: %.3f metri\n', mean_gap_total);