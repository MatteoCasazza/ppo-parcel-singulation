classdef RL_environment < rl.env.MATLABEnvironment
    % RL_environment
    %
    % Reinforcement Learning environment for parcel singulation.
    %
    % System:
    % - 5x5 AMS matrix.
    % - Each AMS has 2 actions:
    %       1) rotation
    %       2) forward velocity
    %
    % Observation:
    % - fixed-size vector 100x1
    % - 25 AMS cells x 4 features
    % - for each cell:
    %       [occupancy, x_normalized, y_normalized, size_normalized]
    %
    % Action:
    % - vector 50x1
    % - normalized in [-1, 1]
    % - internally mapped to:
    %       rotation in [-45 deg, +45 deg]
    %       velocity in [0.5, 2.2] m/s
    %
    % Reward:
    % - inspired by parcel gap reward used in recent literature.
    % - encourages vertical gap between consecutive parcels.
    % - penalizes overlaps and rewards correct singulation.

    %% ================================================================
    %  ENVIRONMENT PROPERTIES
    %% ================================================================

    properties

        %% AMS properties

        %v_treadmill = 0.6;
        %NUOVO: 1.9 m/s dopo indicazioni prof 06/05
         v_treadmill = 1.9;


        d_AMS = 0.2;
        n_i_AMS = 5;
        n_j_AMS = 5;
        n_actions_art = 2;

        upperlim_rot = pi*45/180;
        lowerlim_rot = -pi*45/180;

        upperlim_v = 2.2;
        lowerlim_v = 0.5;

        %% Simulation parameters

        dt = 0.01;
        time = 0;
        cont = 0;

        %% Box parameters
        %Parametro per numero totale di pacchi generato 
         max_gen_boxes = 10;
        % max_gen_boxes = 100; % nuovo 
        n_boxes_tot = 0;

        d_boxes = zeros(10,1);
        x_boxes = zeros(10,1);
        y_boxes = zeros(10,1);

        x_boxes_prec = zeros(10,1);
        y_boxes_prec = zeros(10,1);

        x_boxes_vect = zeros(10,20000);
        y_boxes_vect = zeros(10,20000);

        vy_boxes = zeros(10,1);

        i_box = zeros(10,1);
        j_box = zeros(10,1);
        index_AMS = zeros(10,1);

        pack_exited = zeros(10,1);
        exit_order = zeros(10,1);
        index_exit = 0;
        last_rewarded_exit = 0;

        %% Generation control
        
        %Parametri per frequenza generazione pacchi
        next_box_time = 0.75;
        box_generation_period = 0.75;

        % %NUOVI PER VARI TEST SU FREQUENZA
        % next_box_time = 0.30;
        % box_generation_period = 0.30;

        %Numero di pacchi ad ogni generazione
        % boxes_per_generation = 2;
        %Nuovo per far variare
        boxes_per_generation = 6;

        %% Collision

        toll_contatto = 0.005;

        %% Reward parameters

        desired_gap = 0.15;
        reward_area_length = 1.0;

        %% Last physical action

        LastAction = zeros(50,1);

    end

    properties (Hidden)
        VisualizeAnimation = true
        VisualizeActions = false
        VisualizeStates = false
    end

    properties
        State = zeros(100,1)
    end

    properties(Access = protected)
        IsDone = false
    end

    properties (Transient, Access = private)
        Visualizer = []
    end

    %% ================================================================
    %  CONSTRUCTOR, STEP, RESET
    %% ================================================================

    methods

        function this = RL_environment()

            %% Observation specification
        
            ObservationInfo = rlNumericSpec([100 1]);
        
            ObservationInfo.Name = "System States";
            ObservationInfo.Description = ...
                "25 AMS cells x 4 features: occupancy, normalized x, normalized y, normalized size";
        
            %% Action specification
        
            % 5x5 AMS matrix, each AMS has 2 actions:
            % rotation and velocity
            n_actions = 5 * 5 * 2;
        
            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name = "Normalized AMS Actions";
            ActionInfo.Description = ...
                "Normalized actions in [-1,1]: rotation and velocity for each AMS";
        
            ActionInfo.LowerLimit = -ones(n_actions,1);
            ActionInfo.UpperLimit =  ones(n_actions,1);
        
            %% Create MATLAB RL environment
            % Important:
            % The superclass constructor must be called before using "this".
        
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        
        end

        function [Observation, Reward, IsDone, Info] = step(this, Action)

            Info = [];

            this.time = this.time + this.dt;
            this.cont = this.cont + 1;

            l_AMS_matrix = this.d_AMS * this.n_j_AMS;

            %% --------------------------------------------------------
            %  1. Action denormalization
            %% --------------------------------------------------------

            Action = double(Action(:));
            Action = max(-1, min(1, Action));

            ActLimUp = zeros(50,1);
            ActLimLow = zeros(50,1);

            for kk = 1:2:50

                ActLimLow(kk)   = this.lowerlim_rot;
                ActLimLow(kk+1) = this.lowerlim_v;

                ActLimUp(kk)   = this.upperlim_rot;
                ActLimUp(kk+1) = this.upperlim_v;

            end

            AMS_actions = ActLimLow + (Action + 1) .* (ActLimUp - ActLimLow) ./ 2;
            AMS_actions = max(ActLimLow, min(ActLimUp, AMS_actions));

            this.LastAction = AMS_actions;

            %% --------------------------------------------------------
            %  2. Convert action vector into AMS matrices
            %% --------------------------------------------------------

            rotation_AMS = zeros(5,5);
            v_AMS = zeros(5,5);

            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS

                    idx_rot = jj*2 - 1 + (ii-1)*10;
                    idx_vel = jj*2     + (ii-1)*10;

                    rotation_AMS(ii,jj) = AMS_actions(idx_rot);
                    v_AMS(ii,jj) = AMS_actions(idx_vel);

                end
            end

            %% --------------------------------------------------------
            %  3. Generate new boxes
            %% --------------------------------------------------------

            if this.time >= this.next_box_time && this.n_boxes_tot < this.max_gen_boxes

                this.generateNewBoxes();

                this.next_box_time = this.next_box_time + this.box_generation_period;

            end

            %% --------------------------------------------------------
            %  4. Update parcel kinematics
            %% --------------------------------------------------------

            this.index_AMS = zeros(this.max_gen_boxes,1);

            for kk = 1:this.n_boxes_tot

                %% Bound x before cell detection

                if this.x_boxes(kk) - this.d_boxes(kk)/2 < 0
                    this.x_boxes(kk) = this.d_boxes(kk)/2;
                elseif this.x_boxes(kk) + this.d_boxes(kk)/2 > l_AMS_matrix
                    this.x_boxes(kk) = l_AMS_matrix - this.d_boxes(kk)/2;
                end

                %% Detect current AMS cell

                this.updateBoxCell(kk);

                %% Apply motion

                if this.i_box(kk) <= 5 && this.j_box(kk) <= 5

                    ii = this.i_box(kk);
                    jj = this.j_box(kk);

                    v = v_AMS(ii,jj);
                    r = rotation_AMS(ii,jj);

                    % r = 0 means pure forward motion along y
                    this.x_boxes(kk) = this.x_boxes(kk) + v*this.dt*sin(r);
                    this.y_boxes(kk) = this.y_boxes(kk) + v*this.dt*cos(r);

                    this.vy_boxes(kk) = v*cos(r);

                    this.index_AMS(kk) = jj + (ii-1)*5;

                else

                    % After leaving AMS area, parcels move on treadmill
                    this.y_boxes(kk) = this.y_boxes(kk) + this.v_treadmill*this.dt;
                    this.vy_boxes(kk) = this.v_treadmill;

                    this.index_AMS(kk) = 0;

                end
            end

            %% --------------------------------------------------------
            %  5. Collision handling
            %% --------------------------------------------------------

            this.resolveCollisions();

            %% --------------------------------------------------------
            %  6. Store trajectories
            %% --------------------------------------------------------

            if this.cont <= size(this.x_boxes_vect,2)

                for kk = 1:this.max_gen_boxes

                    if kk <= this.n_boxes_tot

                        this.x_boxes_vect(kk,this.cont) = this.x_boxes(kk);
                        this.y_boxes_vect(kk,this.cont) = this.y_boxes(kk);

                        this.x_boxes_prec(kk) = this.x_boxes(kk);
                        this.y_boxes_prec(kk) = this.y_boxes(kk);

                    else

                        this.x_boxes_vect(kk,this.cont) = -1;
                        this.y_boxes_vect(kk,this.cont) = -1;

                    end
                end
            end

            %% --------------------------------------------------------
            %  7. Check exits and terminal condition
            %% --------------------------------------------------------
            % NUOVO
            % Il traguardo è impostato a 1 metro DOPO l'uscita dell'AMS
            exit_threshold = (this.d_AMS * this.n_i_AMS) + 1.0;

            for kk = 1:this.n_boxes_tot

                if this.y_boxes(kk) > this.d_AMS * this.n_i_AMS

                    if this.pack_exited(kk) == 0

                        this.pack_exited(kk) = 1;

                        this.index_exit = this.index_exit + 1;
                        this.exit_order(this.index_exit) = kk;

                    end
                end
            end

            if this.n_boxes_tot == this.max_gen_boxes && all(this.pack_exited(1:this.max_gen_boxes) == 1)
                IsDone = true;
            else
                IsDone = false;
            end

            this.IsDone = IsDone;

            %% --------------------------------------------------------
            %  8. Observation and reward
            %% --------------------------------------------------------

            Observation = this.getObservation();
            this.State = Observation;

            Reward = this.getReward(AMS_actions);

            notifyEnvUpdated(this);

        end

        function InitialObservation = reset(this)

            %% Reset simulation

            this.time = 0;
            this.cont = 0;

            this.IsDone = false;

            %% Reset boxes

            this.n_boxes_tot = 0;

            this.d_boxes = zeros(this.max_gen_boxes,1);
            this.x_boxes = -ones(this.max_gen_boxes,1);
            this.y_boxes = -ones(this.max_gen_boxes,1);

            this.x_boxes_prec = zeros(this.max_gen_boxes,1);
            this.y_boxes_prec = zeros(this.max_gen_boxes,1);

            this.x_boxes_vect = zeros(this.max_gen_boxes,20000);
            this.y_boxes_vect = zeros(this.max_gen_boxes,20000);

            this.vy_boxes = zeros(this.max_gen_boxes,1);

            this.i_box = zeros(this.max_gen_boxes,1);
            this.j_box = zeros(this.max_gen_boxes,1);
            this.index_AMS = zeros(this.max_gen_boxes,1);

            %% Reset exit info

            this.pack_exited = zeros(this.max_gen_boxes,1);
            this.exit_order = zeros(this.max_gen_boxes,1);
            this.index_exit = 0;
            this.last_rewarded_exit = 0;

            %% Reset generation

            this.next_box_time = this.box_generation_period;

            %% Reset action

            this.LastAction = zeros(50,1);

            %% Generate box sizes

            max_d_box = 2.0 * this.d_AMS;
            min_d_box = 0.25 * this.d_AMS;

            for kk = 1:this.max_gen_boxes
                this.d_boxes(kk) = min_d_box + (max_d_box - min_d_box)*rand();
            end

            %% Generate first two boxes

            this.generateNewBoxes();

            %% Initial observation

            InitialObservation = this.getObservation();
            this.State = InitialObservation;

            notifyEnvUpdated(this);

        end

    end

    %% ================================================================
    %  CUSTOM ENVIRONMENT FUNCTIONS
    %% ================================================================

    methods

        function generateNewBoxes(this)

            if this.n_boxes_tot >= this.max_gen_boxes
                return;
            end

            l_AMS_matrix = this.d_AMS * this.n_j_AMS;

            n_to_generate = min(this.boxes_per_generation, ...
                this.max_gen_boxes - this.n_boxes_tot);

            old_number = this.n_boxes_tot;
            this.n_boxes_tot = this.n_boxes_tot + n_to_generate;

            for local_idx = 1:n_to_generate

                kk = old_number + local_idx;

                if local_idx == 1

                    x_min = this.d_boxes(kk)/2;
                    x_max = this.d_AMS*2.5 - this.d_boxes(kk)/2;

                    x_max = max(x_min, x_max);

                    this.x_boxes(kk) = x_min + (x_max - x_min)*rand();
                    this.y_boxes(kk) = 0.001 + 0.01*rand();

                else

                    x_min = this.d_AMS*2.5 + this.d_boxes(kk)/2;
                    x_max = l_AMS_matrix - this.d_boxes(kk)/2;

                    x_min = min(x_min, x_max);

                    this.x_boxes(kk) = x_min + (x_max - x_min)*rand();
                    this.y_boxes(kk) = 0.001 + 0.05*rand();

                end

                %% Avoid initial overlap with previous generated package

                if kk > 1

                    previous = kk - 1;

                    min_dist = this.d_boxes(kk)/2 + this.d_boxes(previous)/2 + 0.05;

                    if abs(this.x_boxes(kk) - this.x_boxes(previous)) < min_dist

                        if this.x_boxes(kk) >= this.x_boxes(previous)
                            this.x_boxes(kk) = this.x_boxes(previous) + min_dist;
                        else
                            this.x_boxes(kk) = this.x_boxes(previous) - min_dist;
                        end

                    end
                end

                %% Clamp x

                if this.x_boxes(kk) - this.d_boxes(kk)/2 < 0
                    this.x_boxes(kk) = this.d_boxes(kk)/2;
                elseif this.x_boxes(kk) + this.d_boxes(kk)/2 > l_AMS_matrix
                    this.x_boxes(kk) = l_AMS_matrix - this.d_boxes(kk)/2;
                end

                this.x_boxes_prec(kk) = this.x_boxes(kk);
                this.y_boxes_prec(kk) = this.y_boxes(kk);

                this.updateBoxCell(kk);

            end
        end

        function updateBoxCell(this, kk)

            if this.y_boxes(kk) <= this.d_AMS
                this.i_box(kk) = 1;
            elseif this.y_boxes(kk) <= this.d_AMS*2
                this.i_box(kk) = 2;
            elseif this.y_boxes(kk) <= this.d_AMS*3
                this.i_box(kk) = 3;
            elseif this.y_boxes(kk) <= this.d_AMS*4
                this.i_box(kk) = 4;
            elseif this.y_boxes(kk) <= this.d_AMS*5
                this.i_box(kk) = 5;
            else
                this.i_box(kk) = 6;
            end

            if this.i_box(kk) == 6

                this.j_box(kk) = 6;

            elseif this.x_boxes(kk) <= this.d_AMS

                this.j_box(kk) = 1;

            elseif this.x_boxes(kk) <= this.d_AMS*2

                this.j_box(kk) = 2;

            elseif this.x_boxes(kk) <= this.d_AMS*3

                this.j_box(kk) = 3;

            elseif this.x_boxes(kk) <= this.d_AMS*4

                this.j_box(kk) = 4;

            else

                this.j_box(kk) = 5;

            end
        end

        function resolveCollisions(this)

            l_AMS_matrix = this.d_AMS * this.n_j_AMS;

            %% Lateral wall collisions

            for kk = 1:this.n_boxes_tot

                if this.x_boxes(kk) - this.d_boxes(kk)/2 < 0

                    this.x_boxes(kk) = this.d_boxes(kk)/2;

                elseif this.x_boxes(kk) + this.d_boxes(kk)/2 > l_AMS_matrix

                    this.x_boxes(kk) = l_AMS_matrix - this.d_boxes(kk)/2;

                end
            end

            %% Parcel-parcel simple collision resolution

            for a = 1:this.n_boxes_tot
                for b = a+1:this.n_boxes_tot

                    if this.y_boxes(a) < 0 || this.y_boxes(b) < 0
                        continue;
                    end

                    dx = this.x_boxes(b) - this.x_boxes(a);
                    dy = this.y_boxes(b) - this.y_boxes(a);

                    abs_dx = abs(dx);
                    abs_dy = abs(dy);

                    min_dist = this.d_boxes(a)/2 + this.d_boxes(b)/2;

                    %% Resolve only strong overlaps.
                    % The main goal is avoiding unrealistic lateral overlap.
                    % Some longitudinal overlap is tolerated, as in simplified
                    % kinematic sorting models.

                    if abs_dx < min_dist && abs_dy < min_dist

                        overlap_x = min_dist - abs_dx;
                        overlap_y = min_dist - abs_dy;

                        if overlap_x < overlap_y

                            correction = overlap_x/2 + this.toll_contatto;

                            if dx >= 0
                                this.x_boxes(a) = this.x_boxes(a) - correction;
                                this.x_boxes(b) = this.x_boxes(b) + correction;
                            else
                                this.x_boxes(a) = this.x_boxes(a) + correction;
                                this.x_boxes(b) = this.x_boxes(b) - correction;
                            end

                        else

                            correction = overlap_y/2 + this.toll_contatto;

                            if dy >= 0
                                this.y_boxes(a) = this.y_boxes(a) - correction;
                                this.y_boxes(b) = this.y_boxes(b) + correction;
                            else
                                this.y_boxes(a) = this.y_boxes(a) + correction;
                                this.y_boxes(b) = this.y_boxes(b) - correction;
                            end

                        end
                    end
                end
            end

            %% Clamp again after correction

            for kk = 1:this.n_boxes_tot

                if this.x_boxes(kk) - this.d_boxes(kk)/2 < 0

                    this.x_boxes(kk) = this.d_boxes(kk)/2;

                elseif this.x_boxes(kk) + this.d_boxes(kk)/2 > l_AMS_matrix

                    this.x_boxes(kk) = l_AMS_matrix - this.d_boxes(kk)/2;

                end
            end
        end

        function Observation = getObservation(this)

            Observation = zeros(100,1);

            l_AMS_matrix = this.d_AMS * this.n_j_AMS;
            l_AMS_y = this.d_AMS * this.n_i_AMS;

            max_d_box = 2.0 * this.d_AMS;

            %% For each AMS cell, keep the package closest to the exit.
            % This avoids ambiguity if two centroids fall in the same cell.

            cell_occupied = zeros(25,1);
            cell_x = zeros(25,1);
            cell_y = zeros(25,1);
            cell_d = zeros(25,1);
            cell_best_y = -inf(25,1);

            for kk = 1:this.n_boxes_tot

                if this.pack_exited(kk) == 1
                    continue;
                end

                this.updateBoxCell(kk);

                if this.i_box(kk) >= 1 && this.i_box(kk) <= 5 && ...
                        this.j_box(kk) >= 1 && this.j_box(kk) <= 5

                    cell_id = this.j_box(kk) + (this.i_box(kk)-1)*5;

                    if this.y_boxes(kk) > cell_best_y(cell_id)

                        cell_best_y(cell_id) = this.y_boxes(kk);

                        cell_occupied(cell_id) = 1;
                        cell_x(cell_id) = this.x_boxes(kk) / l_AMS_matrix;
                        cell_y(cell_id) = this.y_boxes(kk) / l_AMS_y;
                        cell_d(cell_id) = this.d_boxes(kk) / max_d_box;

                    end
                end
            end

            %% Pack into 100x1 vector

            for cell_id = 1:25

                idx = (cell_id-1)*4 + 1;

                Observation(idx)   = cell_occupied(cell_id);
                Observation(idx+1) = cell_x(cell_id);
                Observation(idx+2) = cell_y(cell_id);
                Observation(idx+3) = cell_d(cell_id);

            end

            Observation = double(Observation(:));

        end
% -----------------------------REWARD FUNCTION------------------------------------
%------------REWARD 9.0------------------------------------
function Reward = getReward(this, AMS_actions)
            Reward = 0;
            this.desired_gap = this.desired_gap;
            
            %% 1. DEFINIZIONE DELLE ZONE (AMS + Treadmill)
            active_idx = [];
            ams_idx = [];
            
            for kk = 1:this.n_boxes_tot
                % Consideriamo TUTTI i pacchi attivi fino alla linea del traguardo (y=2.0)
                if this.pack_exited(kk) == 0 && this.y_boxes(kk) >= 0
                    active_idx = [active_idx; kk];
                    
                    % Isoliamo quelli ancora controllabili fisicamente sull'AMS
                    if this.y_boxes(kk) <= (this.d_AMS * this.n_i_AMS)
                        ams_idx = [ams_idx; kk];
                    end
                end
            end
            
            K = length(active_idx);
            
            if K == 0
                return;
            end
            
            %% 2. INCENTIVO ALL'AVANZAMENTO (Drasticamente ridotto per abbassare il Throughput)
            if ~isempty(ams_idx)
                % Premiamo solo la velocità dei pacchi sull'AMS, e con un peso minimo
                % Questo eviterà che l'agente li spari a 2.2 m/s ignorando i gap.
                mean_vy = mean(this.vy_boxes(ams_idx));
                Reward = Reward + 0.1 * mean_vy; 
            end
            
            %% 3. GRADIENTE DI SINGOLARIZZAZIONE CONTINUO (Esteso su tutti i 2 metri)
            if K >= 2
                y_values = this.y_boxes(active_idx);
                [~, order] = sort(y_values);
                sorted_idx = active_idx(order);
                
                for q = 1:K-1
                    p_back  = sorted_idx(q);
                    p_front = sorted_idx(q+1);
                    
                    y_min_front = this.y_boxes(p_front) - this.d_boxes(p_front)/2;
                    y_max_back  = this.y_boxes(p_back)  + this.d_boxes(p_back)/2;
                    
                    gap = y_min_front - y_max_back;
                    
                    if gap < this.desired_gap
                        % Penalità posizionale proporzionale e smussata
                        gap_error = (this.desired_gap - gap) / this.desired_gap;
                        Reward = Reward - 2.0 * min(gap_error, 2.0);
                        
                        % INCENTIVO DI CORREZIONE (Delta V) solo se il pacco dietro è sull'AMS
                        if this.y_boxes(p_back) <= (this.d_AMS * this.n_i_AMS)
                            vy_front = this.vy_boxes(p_front);
                            vy_back  = this.vy_boxes(p_back);
                            delta_v = vy_front - vy_back;
                            
                            if delta_v > 0
                                Reward = Reward + 2.0 * min(delta_v, 2.0);
                            else
                                Reward = Reward - 2.0 * min(abs(delta_v), 2.0);
                            end
                        end
                    else
                        % Gap perfetto: premio continuo stabilità
                        Reward = Reward + 2.0; 
                    end
                end
            end
            
            %% 4. VALUTAZIONE FINALE ALL'USCITA (Senza "muri" di penalità)
            if this.index_exit > this.last_rewarded_exit
                if this.index_exit >= 2
                    current_box = this.exit_order(this.index_exit);
                    previous_box = this.exit_order(this.index_exit - 1);
        
                    gap_exit = abs(this.y_boxes(current_box) - this.y_boxes(previous_box));
                    gap_exit = gap_exit - this.d_boxes(current_box)/2 - this.d_boxes(previous_box)/2;
        
                    if gap_exit >= this.desired_gap
                        Reward = Reward + 30.0; % Ottimo lavoro KPI Soddisfatto
                    else
                        % CRITICO: Niente più -50 secco. La penalità scala in base a quanto ha sbagliato.
                        % Se sbaglia di 1 cm prenderà -2, se li fa uscire incollati prenderà -30.
                        error_ratio = (this.desired_gap - gap_exit) / this.desired_gap;
                        Reward = Reward - 30.0 * min(error_ratio, 1.5);
                    end
                else
                    Reward = Reward + 5.0; 
                end
                this.last_rewarded_exit = this.index_exit;
            end
            
            %% 5. ACTION PENALTY
            rotations = AMS_actions(1:2:end);
            velocities = AMS_actions(2:2:end);
            rot_effort = mean(abs(rotations) / this.upperlim_rot);
            vel_effort = mean((velocities - this.lowerlim_v) / (this.upperlim_v - this.lowerlim_v));
            Reward = Reward - 0.002*rot_effort - 0.002*vel_effort;
            
            %% Clip di sicurezza
            Reward = double(max(min(Reward, 100), -100));
end
%-------------------FINE REWARD 9.0----------------------------------

        function score = computeFinalSingulationScore(this)

            if this.index_exit < 2
                score = 0;
                return;
            end

            good_gaps = 0;
            total_gaps = this.index_exit - 1;

            for q = 1:total_gaps

                a = this.exit_order(q);
                b = this.exit_order(q+1);

                gap = abs(this.y_boxes(b) - this.y_boxes(a));
                gap = gap - this.d_boxes(a)/2 - this.d_boxes(b)/2;

                %if gap >= this.desired_gap %--> molto stringente come condizione, proviamo con >0 come da indicazioni del prof
                if gap > 0
                    good_gaps = good_gaps + 1;
                end
            end

            score = good_gaps / total_gaps;

        end

    end

    %% ================================================================
    %  VISUALIZATION AND STATE SETTER
    %% ================================================================

    methods

        function varargout = plot(this)

            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                %Da cambiare tra AMSVisualizer e AMSVisualizer_simple a seconda di quello che usi
                this.Visualizer = AMSVisualizer_simple(this);
            else
                bringToFront(this.Visualizer);
            end

            if nargout
                varargout{1} = this.Visualizer;
            end

            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;

        end

        function set.State(this, state)

            validateattributes(state, {'numeric'}, ...
                {'finite','real','vector','numel',100}, '', 'State');

            this.State = double(state(:));

            notifyEnvUpdated(this);

        end

    end

    methods (Access = protected)

        function envUpdatedCallback(this)
            % Optional callback.
        end

    end

end