function [x_best, f_best, convergence, tracking] = GOA_tracking(obj_func, lb, ub, N, T_max, T_B, NY, flags, capture_radius)
% GOA_TRACKING  GOA with per-iteration mechanistic tracking.
%
%  Same algorithm as GOA_fixed, but additionally records:
%    tracking.basins(t)     — distinct basins occupied by 5 Gogessa G6 bests
%    tracking.diversity(t)  — population diversity D(t)
%    tracking.snapshots     — population positions at selected iterations
%    tracking.snapshot_iters — which iterations were snapshotted
%
%  Used for the mechanistic analysis section of the manuscript.

    if nargin < 4  || isempty(N),     N     = 50;  end
    if nargin < 5  || isempty(T_max), T_max = 500; end
    if nargin < 6  || isempty(T_B),   T_B   = 25;  end
    if nargin < 7  || isempty(NY),    NY    = 10;  end
    if nargin < 8  || isempty(flags)
        flags.use_butta=true; flags.use_yuba=true; flags.use_topology=true;
        flags.use_siinqee=true; flags.use_gumii=true; flags.use_incremental=true;
        flags.use_g5shadow=true; flags.beta_val=0.1;
    end
    if ~isfield(flags,'use_incremental'), flags.use_incremental = true; end
    if ~isfield(flags,'use_g5shadow'),    flags.use_g5shadow    = true; end
    if ~isfield(flags,'beta_val'),        flags.beta_val        = 0.1;  end
    if nargin < 9 || isempty(capture_radius), capture_radius = 1.0; end

    K           = 5;
    d           = numel(lb);
    grade_sizes = [2,1,1,2,2,2];
    N_actual    = K * sum(grade_sizes);

    safe_eval = @(x) eval_safe(x, obj_func);

    %% Tracking setup
    tracking.basins    = zeros(T_max, 1);
    tracking.diversity = zeros(T_max, 1);
    tracking.gogessa_basins = zeros(T_max, K);  % per-Gogessa G6 best positions

    % Snapshot iterations
    snapshot_iters = unique([1, 25, 50, 100, 150, 250, T_B, 2*T_B, ...
                            floor(T_max/2), T_max]);
    snapshot_iters = sort(snapshot_iters(snapshot_iters >= 1 & snapshot_iters <= T_max));
    tracking.snapshot_iters = snapshot_iters;
    tracking.snapshots = cell(numel(snapshot_iters), 1);

    %% Initialisation (identical to GOA_fixed)
    pop = struct();
    for k = 1:K
        for g = 1:6
            ng = grade_sizes(g);
            pop(k).grade{g}    = lb + rand(ng,d).*(ub-lb);
            pop(k).grade_f{g}  = zeros(ng,1);
            pop(k).grade_cv{g} = zeros(ng,1);
            for j = 1:ng
                [pop(k).grade_f{g}(j), pop(k).grade_cv{g}(j)] = safe_eval(pop(k).grade{g}(j,:));
            end
        end
    end

    Y_archive = []; Y_fitness = []; Y_cv = [];

    f_best = inf; cv_best = inf; x_best = pop(1).grade{6}(1,:);
    for k = 1:K
        for j = 1:size(pop(k).grade{6},1)
            if deb_compare(pop(k).grade_f{6}(j), pop(k).grade_cv{6}(j), f_best, cv_best)
                f_best  = pop(k).grade_f{6}(j);
                cv_best = pop(k).grade_cv{6}(j);
                x_best  = pop(k).grade{6}(j,:);
            end
        end
    end

    convergence = zeros(T_max,1);
    delta_alpha = 1.0;

    %% Main loop
    for t = 1:T_max

        alpha_t = max(2*(1 - t/T_max)*delta_alpha, 0.01);

        if flags.use_yuba, Y_arc_pass = Y_archive; else, Y_arc_pass = []; end

        pop = GOA_grades(pop, x_best, Y_arc_pass, alpha_t, K, lb, ub, ...
                         safe_eval, flags.use_incremental, ...
                         flags.beta_val, flags.use_g5shadow);

        % Ally sharing
        if flags.use_topology && mod(t,4)==0
            ally_pairs = [1,3; 2,4; 3,5];
            for ap = 1:size(ally_pairs,1)
                k1 = ally_pairs(ap,1); k2 = ally_pairs(ap,2);
                [x_k1,~] = gogessa_best_local(pop, k1);
                [x_k2,~] = gogessa_best_local(pop, k2);
                for j = 1:size(pop(k2).grade{5},1)
                    x_new = max(lb,min(ub, pop(k2).grade{5}(j,:) + 0.15*(x_k1-pop(k2).grade{5}(j,:))));
                    [f_new,cv_new] = safe_eval(x_new);
                    if deb_compare(f_new,cv_new,pop(k2).grade_f{5}(j),pop(k2).grade_cv{5}(j))
                        pop(k2).grade{5}(j,:)=x_new; pop(k2).grade_f{5}(j)=f_new; pop(k2).grade_cv{5}(j)=cv_new;
                    end
                end
                for j = 1:size(pop(k1).grade{5},1)
                    x_new = max(lb,min(ub, pop(k1).grade{5}(j,:) + 0.15*(x_k2-pop(k1).grade{5}(j,:))));
                    [f_new,cv_new] = safe_eval(x_new);
                    if deb_compare(f_new,cv_new,pop(k1).grade_f{5}(j),pop(k1).grade_cv{5}(j))
                        pop(k1).grade{5}(j,:)=x_new; pop(k1).grade_f{5}(j)=f_new; pop(k1).grade_cv{5}(j)=cv_new;
                    end
                end
            end
        end

        % Collect all agents
        all_x=[]; all_f=[]; all_cv=[];
        for k=1:K
            for g=1:6
                all_x =[all_x; pop(k).grade{g}];
                all_f =[all_f; pop(k).grade_f{g}];
                all_cv=[all_cv;pop(k).grade_cv{g}];
            end
        end

        x_mean = mean(all_x,1);
        D_t    = mean(vecnorm(all_x-x_mean,2,2));
        D_max  = 0.3*norm(ub-lb);
        D_min  = 0.001*norm(ub-lb);

        % Siinqee
        if flags.use_siinqee && (D_t/(D_max+eps)) < 0.05
            n_perturb = max(1, round(0.1*N_actual));
            [~,best_idx] = min(all_f + 1e12*(all_cv>1e-10));
            cand_idx = setdiff(1:N_actual, best_idx);
            sel_idx  = cand_idx(randperm(numel(cand_idx), min(n_perturb,numel(cand_idx))));
            for si = 1:numel(sel_idx)
                idx = sel_idx(si);
                sigma_S = 0.1*(all_f(idx)-f_best)/(abs(f_best)+eps)*norm(ub-lb);
                sigma_S = max(sigma_S, 1e-6);
                x_pert = max(lb,min(ub, all_x(idx,:)+sigma_S*randn(1,d)));
                [f_pert,cv_pert] = safe_eval(x_pert);
                if deb_compare(f_pert,cv_pert,all_f(idx),all_cv(idx))
                    all_x(idx,:)=x_pert; all_f(idx)=f_pert; all_cv(idx)=cv_pert;
                end
            end
            row=0;
            for k=1:K
                for g=1:6
                    ng=size(pop(k).grade{g},1);
                    for j=1:ng
                        row=row+1;
                        pop(k).grade{g}(j,:)  =all_x(row,:);
                        pop(k).grade_f{g}(j)  =all_f(row);
                        pop(k).grade_cv{g}(j) =all_cv(row);
                    end
                end
            end
        end

        % Gumii Gayo
        if flags.use_gumii && mod(t,T_B)==floor(T_B/2)
            if D_t < D_min,      delta_alpha = min(delta_alpha*1.2, 2.0);
            elseif D_t > D_max,  delta_alpha = delta_alpha*0.85;
            end
        end

        % Butta
        if flags.use_butta && mod(t,T_B)==0
            [pop,Y_archive,Y_fitness,Y_cv,x_best,f_best,cv_best] = ...
                GOA_butta_deb(pop,Y_archive,Y_fitness,Y_cv, ...
                              x_best,f_best,cv_best,lb,ub,K,NY,safe_eval);
        end

        % Update global best
        for k=1:K
            for g=1:6
                for j=1:size(pop(k).grade{g},1)
                    if deb_compare(pop(k).grade_f{g}(j),pop(k).grade_cv{g}(j),f_best,cv_best)
                        f_best=pop(k).grade_f{g}(j); cv_best=pop(k).grade_cv{g}(j);
                        x_best=pop(k).grade{g}(j,:);
                    end
                end
            end
        end

        convergence(t) = f_best;

        %% ====== TRACKING ======

        % 1. Basin coverage: G6 best per Gogessa
        reps = zeros(K, d);
        for k = 1:K
            g6_f = pop(k).grade_f{6};
            [~, bi] = min(g6_f);
            reps(k,:) = pop(k).grade{6}(bi,:);
        end
        tracking.basins(t) = count_basins_local(reps, capture_radius);

        % 2. Diversity
        all_x2 = [];
        for k=1:K
            for g=1:6
                all_x2 = [all_x2; pop(k).grade{g}]; %#ok<AGROW>
            end
        end
        xm = mean(all_x2, 1);
        tracking.diversity(t) = mean(vecnorm(all_x2 - xm, 2, 2));

        % 3. Snapshots
        snap_idx = find(snapshot_iters == t);
        if ~isempty(snap_idx)
            snap = struct();
            snap.all_x = all_x2;
            snap.gogessa_labels = [];
            snap.grade_labels = [];
            for k = 1:K
                for g = 1:6
                    ng = size(pop(k).grade{g}, 1);
                    snap.gogessa_labels = [snap.gogessa_labels; k*ones(ng,1)];
                    snap.grade_labels   = [snap.grade_labels; g*ones(ng,1)];
                end
            end
            snap.reps = reps;
            snap.x_best = x_best;
            snap.f_best = f_best;
            tracking.snapshots{snap_idx} = snap;
        end
    end
end


function n = count_basins_local(reps, r)
    K = size(reps, 1);
    assigned = false(K, 1);
    n = 0;
    for i = 1:K
        if assigned(i), continue; end
        n = n + 1;
        assigned(i) = true;
        for j = i+1:K
            if ~assigned(j) && norm(reps(i,:) - reps(j,:)) < r
                assigned(j) = true;
            end
        end
    end
end


function [x_gb,f_gb] = gogessa_best_local(pop,k)
    f_gb=inf; cv_gb=inf; x_gb=pop(k).grade{6}(1,:);
    for j=1:size(pop(k).grade{6},1)
        if deb_compare(pop(k).grade_f{6}(j),pop(k).grade_cv{6}(j),f_gb,cv_gb)
            f_gb=pop(k).grade_f{6}(j); cv_gb=pop(k).grade_cv{6}(j);
            x_gb=pop(k).grade{6}(j,:);
        end
    end
end
