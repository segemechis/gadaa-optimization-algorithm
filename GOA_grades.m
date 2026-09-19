function [pop, S_F, S_CR] = GOA_grades(pop, x_best, Y_archive, alpha_t, K, lb, ub, ...
                                        safe_eval, use_incremental, beta, use_g5shadow, ...
                                        M_F, M_CR, H, all_x_global, t, T_max, use_gamme_de)

    if nargin < 9,  use_incremental = false; end
    if nargin < 10, beta = 0.1; end
    if nargin < 11, use_g5shadow = true; end
    if nargin < 12 || isempty(M_F),  M_F  = 0.5*ones(5,1); end
    if nargin < 13 || isempty(M_CR), M_CR = 0.5*ones(5,1); end
    if nargin < 14 || isempty(H),    H    = 5; end
    if nargin < 15, all_x_global = []; end
    if nargin < 16, t = 1; end
    if nargin < 17, T_max = 500; end
    if nargin < 18 || isempty(use_gamme_de), use_gamme_de = true; end

    d  = numel(lb);
    NY = size(Y_archive, 1);

    % [P2] Preallocate S_F / S_CR
    max_success = K * 4 * 2;  % conservative upper bound
    S_F  = zeros(max_success, 1); sf_count = 0;
    S_CR = zeros(max_success, 1); scr_count = 0;

    % [P2] OBL gate: only run G6 OBL in first 70% of budget
    run_obl = (t < 0.7 * T_max);

    for k = 1:K

        % Gogessa centroid and flat view
        all_x_k = []; all_f_k = [];
        for g = 1:6
            all_x_k = [all_x_k; pop(k).grade{g}];   %#ok<AGROW>
            all_f_k = [all_f_k; pop(k).grade_f{g}];  %#ok<AGROW>
        end
        x_centroid = mean(all_x_k, 1);
        range = ub - lb;

        % Mutation pool: all_x_global already contains all agents (no local dup)
        mut_pool = all_x_global;
        if ~isempty(Y_archive), mut_pool = [mut_pool; Y_archive]; end
        n_pool = size(mut_pool, 1);

        % p-best: top p% of LOCAL Gogessa only (fitness known, safe)
        p_rate       = 0.1;
        n_local_pool = size(all_x_k, 1);
        n_pbest      = max(2, ceil(p_rate * n_local_pool));
        [~, pbest_sort] = sort(all_f_k);
        pbest_idx_set   = pbest_sort(1:n_pbest);  % indices into all_x_k

        % -------------------------------------------------------------------
        % G6 -- Gada: Abba Gadaa attraction + OBL [P2: OBL gated by t]
        % -------------------------------------------------------------------
        if NY > 0
            p_idx = min(ceil(k*NY/K), NY);
            y_k   = Y_archive(p_idx,:) - pop(k).grade{6}(1,:);
        else
            y_k = zeros(1,d);
        end

        n6 = size(pop(k).grade{6}, 1);
        for j = 1:n6
            x_old = pop(k).grade{6}(j,:);
            alpha_eff = max(alpha_t, 0.01);
            x_new = x_old + alpha_eff*(x_best - x_old) + beta*y_k;
            x_new = reflect_bounds(x_new, lb, ub);
            [f_new, cv_new] = safe_eval(x_new);
            if deb_compare(f_new,cv_new,pop(k).grade_f{6}(j),pop(k).grade_cv{6}(j))
                pop(k).grade{6}(j,:)  = x_new;
                pop(k).grade_f{6}(j)  = f_new;
                pop(k).grade_cv{6}(j) = cv_new;
            end

            % [P2] OBL only in first 70% of budget
            if run_obl
                x_opp = max(lb, min(ub, lb + ub - pop(k).grade{6}(j,:)));
                [f_opp, cv_opp] = safe_eval(x_opp);
                if deb_compare(f_opp,cv_opp,pop(k).grade_f{6}(j),pop(k).grade_cv{6}(j))
                    pop(k).grade{6}(j,:)  = x_opp;
                    pop(k).grade_f{6}(j)  = f_opp;
                    pop(k).grade_cv{6}(j) = cv_opp;
                end
            end
        end

        % -------------------------------------------------------------------
        % G5 -- Raba Dori: Gobo learning + Covariance-guided refinement
        % -------------------------------------------------------------------
        g6_best_f=inf; g6_best_cv=inf; g6_best_idx=1;
        for jj=1:size(pop(k).grade{6},1)
            if deb_compare(pop(k).grade_f{6}(jj),pop(k).grade_cv{6}(jj),g6_best_f,g6_best_cv)
                g6_best_f=pop(k).grade_f{6}(jj); g6_best_cv=pop(k).grade_cv{6}(jj);
                g6_best_idx=jj;
            end
        end
        x_g6_best = pop(k).grade{6}(g6_best_idx,:);

        % Build local covariance with [P2] Tikhonov regularization
        local_pop = [pop(k).grade{5}; pop(k).grade{6}];
        n_local = size(local_pop,1);
        use_cov = (n_local > d);
        if use_cov
            try
                C_local = cov(local_pop);
                % [P2] Tikhonov regularization: prevents rank-deficiency for small n_local
                C_local = C_local + 1e-6 * eye(d) * trace(C_local) / d;
                [B_eig, D_eig] = eig(C_local);
                D_eig = max(real(diag(D_eig)), 1e-20);
                D_eig = sqrt(D_eig);
                B_eig = real(B_eig);
            catch
                use_cov = false;
            end
        end

        n5 = size(pop(k).grade{5}, 1);
        for j = 1:n5
            % Phase 1: Gobo learning
            r1=rand; r2=rand;
            x_old = pop(k).grade{5}(j,:);
            x_new = x_old + 0.7*r1*(x_g6_best-x_old) + 0.3*r2*(x_best-x_old);
            x_new = reflect_bounds(x_new, lb, ub);
            [f_new,cv_new] = safe_eval(x_new);
            if deb_compare(f_new,cv_new,pop(k).grade_f{5}(j),pop(k).grade_cv{5}(j))
                pop(k).grade{5}(j,:)  = x_new;
                pop(k).grade_f{5}(j)  = f_new;
                pop(k).grade_cv{5}(j) = cv_new;
            end

            % Phase 2: Covariance-guided refinement
            if use_g5shadow
                x_loc  = pop(k).grade{5}(j,:);
                f_loc  = pop(k).grade_f{5}(j);
                cv_loc = pop(k).grade_cv{5}(j);
                sigma_loc = 0.02*norm(ub-lb)*max(alpha_t,0.01);
                for trial = 1:5
                    if use_cov
                        z = randn(d,1);
                        y = B_eig*(D_eig.*z);
                        x_try = x_loc + sigma_loc*y';
                    else
                        x_try = x_loc + sigma_loc*randn(1,d);
                    end
                    x_try = reflect_bounds(x_try, lb, ub);
                    [f_try,cv_try] = safe_eval(x_try);
                    if deb_compare(f_try,cv_try,f_loc,cv_loc)
                        x_loc=x_try; f_loc=f_try; cv_loc=cv_try;
                        sigma_loc=sigma_loc*1.5;
                    else
                        sigma_loc=sigma_loc*0.7;
                    end
                end
                pop(k).grade{5}(j,:)  = x_loc;
                pop(k).grade_f{5}(j)  = f_loc;
                pop(k).grade_cv{5}(j) = cv_loc;
            end
        end

        % -------------------------------------------------------------------
        % G4 -- Kusa: dual-move vetting
        % -------------------------------------------------------------------
        [~,k_best_idx] = min(all_f_k);
        x_k_best = all_x_k(k_best_idx,:);
        n_total  = size(all_x_k,1);
        n4 = size(pop(k).grade{4},1);
        for j=1:n4
            r1=rand; r2=rand;
            x_old=pop(k).grade{4}(j,:);
            x_rand=all_x_k(randi(n_total),:);
            c1=reflect_bounds(x_old+0.6*r1*(x_k_best-x_old),lb,ub);
            c2=reflect_bounds(x_old+0.4*r2*(x_rand-x_old),lb,ub);
            [f1,cv1]=safe_eval(c1); [f2,cv2]=safe_eval(c2);
            f0=pop(k).grade_f{4}(j); cv0=pop(k).grade_cv{4}(j);
            if deb_compare(f1,cv1,f2,cv2)&&deb_compare(f1,cv1,f0,cv0)
                pop(k).grade{4}(j,:)=c1; pop(k).grade_f{4}(j)=f1; pop(k).grade_cv{4}(j)=cv1;
            elseif deb_compare(f2,cv2,f1,cv1)&&deb_compare(f2,cv2,f0,cv0)
                pop(k).grade{4}(j,:)=c2; pop(k).grade_f{4}(j)=f2; pop(k).grade_cv{4}(j)=cv2;
            end
        end

        % -------------------------------------------------------------------
        % G3 & G2 -- Gamme: DE/current-to-pbest/1 with SHADE parameters
        %   x_pbest from top p% of local Gogessa (pbest_idx_set computed above)
        % -------------------------------------------------------------------
        % G2-G3 use pbest_idx_set computed at top of Gogessa loop above
        for gi=1:2
            g=gi+1;
            n_g=size(pop(k).grade{g},1);
            for j=1:n_g
                x_old  = pop(k).grade{g}(j,:);
                f_old  = pop(k).grade_f{g}(j);
                cv_old = pop(k).grade_cv{g}(j);

                if use_gamme_de
                    % --- Gamme Differential Update (DE-based, rotationally invariant) ---
                    r_h = randi(H);

                    % [P1 FIX] Resample F until > 0
                    F_i = M_F(r_h) + 0.1*tan(pi*(rand-0.5));
                    while F_i <= 0
                        F_i = M_F(r_h) + 0.1*tan(pi*(rand-0.5));
                    end
                    F_i = min(F_i, 1.0);

                    CR_i = min(1, max(0, M_CR(r_h) + 0.1*randn));

                    % [Strategic] Sample x_pbest from top p% of local Gogessa
                    pb_pick = pbest_idx_set(randi(n_pbest));
                    x_pbest = all_x_k(pb_pick,:);  % from local agents only

                    % Mutation parents (distinct from each other and from x_old)
                    if n_pool >= 3
                        idx3 = randperm(n_pool, 3);
                        xr1  = mut_pool(idx3(1),:);
                        xr2  = mut_pool(idx3(2),:);
                    else
                        xr1 = lb + rand(1,d).*(ub-lb);
                        xr2 = lb + rand(1,d).*(ub-lb);
                    end

                    % DE/current-to-pbest/1
                    x_mut = x_old + F_i*(x_pbest-x_old) + F_i*(xr1-xr2);
                    x_mut = reflect_bounds(x_mut, lb, ub);

                    % Binomial crossover
                    j_rand = randi(d);
                    mask   = rand(1,d) < CR_i; mask(j_rand)=true;
                    x_trial = x_old; x_trial(mask) = x_mut(mask);
                    x_trial = reflect_bounds(x_trial, lb, ub);

                    [f_t,cv_t] = safe_eval(x_trial);
                    if deb_compare(f_t,cv_t,f_old,cv_old)
                        pop(k).grade{g}(j,:)  = x_trial;
                        pop(k).grade_f{g}(j)  = f_t;
                        pop(k).grade_cv{g}(j) = cv_t;
                        sf_count=sf_count+1; S_F(sf_count)  = F_i;
                        scr_count=scr_count+1; S_CR(scr_count)= CR_i;
                    end

                else
                    % --- Centroid Fallback: simple step, axis-aligned, no DE ---
                    % Tests what GOA loses without rotational invariance in G2-G3
                    x_trial = x_old + alpha_t*(x_centroid - x_old) ...
                            + (0.08/sqrt(d))*(range.*(2*rand(1,d)-1));
                    x_trial = reflect_bounds(x_trial, lb, ub);
                    [f_t,cv_t] = safe_eval(x_trial);
                    if deb_compare(f_t,cv_t,f_old,cv_old)
                        pop(k).grade{g}(j,:)  = x_trial;
                        pop(k).grade_f{g}(j)  = f_t;
                        pop(k).grade_cv{g}(j) = cv_t;
                        % No S_F/S_CR recording — centroid has no DE parameters
                    end
                end  % use_gamme_de

            end  % j
        end  % gi

        % -------------------------------------------------------------------
        % G1 -- Dabale: [P0 FIX] Evaluate EVERY iteration (not just Butta)
        % -------------------------------------------------------------------
        n1 = size(pop(k).grade{1}, 1);
        for j=1:n1
            r   = rand; u = -1+2*rand(1,d);
            x_old = pop(k).grade{1}(j,:);
            if use_incremental
                x_new = x_old + 0.8*r*(x_centroid-x_old) + (0.08/sqrt(d))*(range.*u);
            else
                x_new = x_centroid + 0.8*r*(range.*u);
            end
            x_new = reflect_bounds(x_new, lb, ub);
            % [P0] EVALUATE immediately — no wasted FE budget
            [f_new, cv_new] = safe_eval(x_new);
            if deb_compare(f_new,cv_new,pop(k).grade_f{1}(j),pop(k).grade_cv{1}(j))
                pop(k).grade{1}(j,:)  = x_new;
                pop(k).grade_f{1}(j)  = f_new;
                pop(k).grade_cv{1}(j) = cv_new;
            end
        end

        % -------------------------------------------------------------------
        % Abbaa Dula: scouting raid
        % -------------------------------------------------------------------
        k_opp=mod(k,K)+1;
        [~,g6_worst_idx]=max(pop(k).grade_f{6});
        x_worst=pop(k).grade{6}(g6_worst_idx,:);
        [~,opp_best_idx]=min(pop(k_opp).grade_f{6});
        x_opp_best=pop(k_opp).grade{6}(opp_best_idx,:);
        eta=rand; xi=0.05*norm(ub-lb);
        x_raid=x_worst+eta*(x_opp_best-x_worst)+xi*(-1+2*rand(1,d));
        x_raid=reflect_bounds(x_raid,lb,ub);
        [f_raid,cv_raid]=safe_eval(x_raid);
        if deb_compare(f_raid,cv_raid,pop(k).grade_f{6}(g6_worst_idx),pop(k).grade_cv{6}(g6_worst_idx))
            pop(k).grade{6}(g6_worst_idx,:)=x_raid;
            pop(k).grade_f{6}(g6_worst_idx)=f_raid;
            pop(k).grade_cv{6}(g6_worst_idx)=cv_raid;
        end

    end  % Gogessa loop

    % Trim preallocated arrays
    S_F  = S_F(1:sf_count);
    S_CR = S_CR(1:scr_count);
end


function x = reflect_bounds(x, lb, ub)
    below=x<lb; above=x>ub;
    x(below)=2*lb(below)-x(below);
    x(above)=2*ub(above)-x(above);
    x=max(lb,min(ub,x));
end
