function [x_best, f_best, convergence, pop_final, goa_data] = GOA_fixed(obj_func, lb, ub, N, T_max, T_B, NY, flags)

    if nargin<4||isempty(N),     N    =50;  end
    if nargin<5||isempty(T_max), T_max=500; end
    if nargin<6||isempty(T_B),   T_B  =25;  end
    if nargin<7||isempty(NY),    NY   =10;  end
    if nargin<8||isempty(flags)
        flags.use_butta=true; flags.use_yuba=true; flags.use_topology=true;
        flags.use_siinqee=true; flags.use_gumii=true;
        flags.use_incremental=true; flags.use_g5shadow=true; flags.beta_val=0.1;
    end
    if ~isfield(flags,'use_incremental'), flags.use_incremental=true; end
    if ~isfield(flags,'use_g5shadow'),    flags.use_g5shadow=true;    end
    if ~isfield(flags,'beta_val'),        flags.beta_val=0.1;         end
    if ~isfield(flags,'problem_type'),    flags.problem_type='static'; end
    if ~isfield(flags,'use_apm'),         flags.use_apm=true;         end
    if ~isfield(flags,'use_eigen'),       flags.use_eigen=true;       end
    if ~isfield(flags,'use_gamme_de'),    flags.use_gamme_de=true;    end
    if ~isfield(flags,'use_nlpsr'),       flags.use_nlpsr=true;       end

    K           = 5;
    K_min       = 2;      % [P0] NLPSR minimum Gogessas
    d           = numel(lb);
    grade_sizes = [2,1,1,2,2,2];
    N_actual    = K*sum(grade_sizes);  % 50 (based on full K)

    safe_eval = @(x) eval_safe(x, obj_func);

    D_max   = 0.3  *norm(ub-lb);
    D_min   = 0.001*norm(ub-lb);
    sigma_Y = 0.05 *norm(ub-lb);

    % [K1] 
    H_mem=10; M_F=0.5*ones(H_mem,1); M_CR=0.5*ones(H_mem,1); k_mem=1;

    %% Initialisation — full K=5 Gogessas
    pop=struct();
    for k=1:K
        for g=1:6
            ng=grade_sizes(g);
            pop(k).grade{g}    = lb+rand(ng,d).*(ub-lb);
            pop(k).grade_f{g}  = zeros(ng,1);
            pop(k).grade_cv{g} = zeros(ng,1);
            for j=1:ng
                [pop(k).grade_f{g}(j),pop(k).grade_cv{g}(j)]=safe_eval(pop(k).grade{g}(j,:));
            end
        end
    end

    Y_archive=[]; Y_fitness=[]; Y_cv=[];
    f_best=inf; cv_best=inf; x_best=pop(1).grade{6}(1,:);
    for k=1:K
        for j=1:size(pop(k).grade{6},1)
            if deb_compare(pop(k).grade_f{6}(j),pop(k).grade_cv{6}(j),f_best,cv_best)
                f_best=pop(k).grade_f{6}(j); cv_best=pop(k).grade_cv{6}(j);
                x_best=pop(k).grade{6}(j,:);
            end
        end
    end

    convergence=zeros(T_max,1); delta_alpha=1.0;
    last_butta_t=0; butta_min_interval=max(10,round(T_B*0.6));

    %% Tracking data struct (5th output) — for figure generation
    goa_data.diversity_curve   = zeros(T_max,1); % population diversity D_t
    goa_data.avg_fitness_curve = zeros(T_max,1); % mean agent fitness per iter
    goa_data.explore_rate      = zeros(T_max,1); % D_t/D_max (exploration proxy)
    goa_data.exploit_rate      = zeros(T_max,1); % 1-D_t/D_max (exploitation proxy)
    goa_data.xbest_history     = zeros(T_max,d); % x_best position per iter
    goa_data.butta_iters       = [];             % iterations when Butta triggered
    goa_data.K_history         = zeros(T_max,1); % K_current per iter

    %% Main loop
    for t=1:T_max

        alpha_t=max(2*(1-t/T_max)*delta_alpha, 0.01);

        %% [P0] NLPSR — gentler non-linear reduction
        % Shrinks slowly first half, accelerates in final phase
        if strcmp(flags.problem_type,'static') && flags.use_nlpsr
            K_current = max(K_min, round(K*(1-(t/T_max)^(1-t/T_max))));
        else
            K_current = K;  % no reduction for dynamic problems OR use_nlpsr=false
        end

        if flags.use_yuba, Y_arc_pass=Y_archive; else, Y_arc_pass=[]; end

        % Collect global pop from ALL K Gogessas (mutation parents)
        all_x_global=[];
        for k=1:K
            for g=1:6
                all_x_global=[all_x_global; pop(k).grade{g}]; %#ok<AGROW>
            end
        end

        %% Grade updates — only active K_current Gogessas
        [pop, S_F, S_CR] = GOA_grades(pop, x_best, Y_arc_pass, alpha_t, K_current, lb, ub, ...
                                       safe_eval, flags.use_incremental, ...
                                       flags.beta_val, flags.use_g5shadow, ...
                                       M_F, M_CR, H_mem, all_x_global, t, T_max, flags.use_gamme_de);

        %% [K1] SHADE update — Lehmer for M_F, arithmetic mean for M_CR
        if flags.use_apm && numel(S_F)>0
            M_F(k_mem)=sum(S_F.^2)/(sum(S_F)+eps);    % Lehmer mean (correct for F)
            if numel(S_CR)>0
                M_CR(k_mem)=mean(S_CR);                % Arithmetic mean (correct for CR)
            end
            k_mem=mod(k_mem,H_mem)+1;
        end

        %% [K4] Eigenvector crossover every 50 iterations — correct math
        if flags.use_eigen && mod(t,50)==0
            all_f_flat=[]; all_cv_flat=[];
            for k=1:K
                for g=1:6
                    all_f_flat =[all_f_flat;  pop(k).grade_f{g}];  %#ok<AGROW>
                    all_cv_flat=[all_cv_flat; pop(k).grade_cv{g}]; %#ok<AGROW>
                end
            end
            n_elite=max(d+1,floor(N_actual/2));
            [~,sort_idx]=sort(all_f_flat+1e12*(all_cv_flat>1e-10));
            elite_x=all_x_global(sort_idx(1:min(n_elite,numel(sort_idx))),:);
            if size(elite_x,1)>d
                try
                    mu_elite=mean(elite_x,1);
                    C_eig=cov(elite_x);
                    [V_eig,D_eig_mat]=eig(C_eig);
                    eigvals=max(real(diag(D_eig_mat)),1e-12);
                    V_eig=real(V_eig); eig_ok=true;
                catch, eig_ok=false; end
                if eig_ok
                    sigma_eig=0.05*norm(ub-lb)*max(alpha_t,0.01);
                    for k=1:K_current
                        for g=5:6
                            for j=1:size(pop(k).grade{g},1)
                                z=V_eig'*(pop(k).grade{g}(j,:)'-mu_elite');
                                z_new=z+sigma_eig*sqrt(eigvals).*randn(d,1);
                                x_try=mu_elite+(V_eig*z_new)';
                                x_try=max(lb,min(ub,x_try));
                                [f_try,cv_try]=safe_eval(x_try);
                                if deb_compare(f_try,cv_try,pop(k).grade_f{g}(j),pop(k).grade_cv{g}(j))
                                    pop(k).grade{g}(j,:)=x_try;
                                    pop(k).grade_f{g}(j)=f_try;
                                    pop(k).grade_cv{g}(j)=cv_try;
                                end
                            end
                        end
                    end
                end
            end
        end

        %% 
        if flags.use_topology&&mod(t,4)==0
            ally_pairs=[1,3;2,4;3,5];
            for ap=1:size(ally_pairs,1)
                k1=ally_pairs(ap,1); k2=ally_pairs(ap,2);
                if k1>K_current || k2>K_current, continue; end
                % Both guaranteed ≤ K_current — no min() clamping needed
                [x_k1,~]=gogessa_best_local(pop,k1);
                [x_k2,~]=gogessa_best_local(pop,k2);
                F_ally=M_F(randi(H_mem))+0.1*tan(pi*(rand-0.5));
                while F_ally<=0, F_ally=M_F(randi(H_mem))+0.1*tan(pi*(rand-0.5)); end
                F_ally=min(1,F_ally);
                for j=1:size(pop(k2).grade{5},1)
                    x_new=max(lb,min(ub,pop(k2).grade{5}(j,:)+F_ally*(x_k1-pop(k2).grade{5}(j,:))));
                    [f_new,cv_new]=safe_eval(x_new);
                    if deb_compare(f_new,cv_new,pop(k2).grade_f{5}(j),pop(k2).grade_cv{5}(j))
                        pop(k2).grade{5}(j,:)=x_new; pop(k2).grade_f{5}(j)=f_new; pop(k2).grade_cv{5}(j)=cv_new;
                    end
                end
                F_ally=M_F(randi(H_mem))+0.1*tan(pi*(rand-0.5));
                while F_ally<=0, F_ally=M_F(randi(H_mem))+0.1*tan(pi*(rand-0.5)); end
                F_ally=min(1,F_ally);
                for j=1:size(pop(k1).grade{5},1)
                    x_new=max(lb,min(ub,pop(k1).grade{5}(j,:)+F_ally*(x_k2-pop(k1).grade{5}(j,:))));
                    [f_new,cv_new]=safe_eval(x_new);
                    if deb_compare(f_new,cv_new,pop(k1).grade_f{5}(j),pop(k1).grade_cv{5}(j))
                        pop(k1).grade{5}(j,:)=x_new; pop(k1).grade_f{5}(j)=f_new; pop(k1).grade_cv{5}(j)=cv_new;
                    end
                end
            end
        end

        %% Collect all agents (all K, not just K_current)
        all_x=[]; all_f=[]; all_cv=[];
        for k=1:K
            for g=1:6
                all_x =[all_x;  pop(k).grade{g}];    %#ok<AGROW>
                all_f =[all_f;  pop(k).grade_f{g}];  %#ok<AGROW>
                all_cv=[all_cv; pop(k).grade_cv{g}]; %#ok<AGROW>
            end
        end
        x_mean=mean(all_x,1); D_t=mean(vecnorm(all_x-x_mean,2,2));

        %% Siinqee
        if flags.use_siinqee&&(D_t/(D_max+eps))<0.05
            N_actual_curr=K_current*sum(grade_sizes);
            n_perturb=max(1,round(0.1*N_actual_curr));
            [~,best_idx]=min(all_f+1e12*(all_cv>1e-10));
            cand_idx=setdiff(1:size(all_x,1),best_idx);
            sel_idx=cand_idx(randperm(numel(cand_idx),min(n_perturb,numel(cand_idx))));
            for si=1:numel(sel_idx)
                idx=sel_idx(si);
                sigma_S=0.1*(all_f(idx)-f_best)/(abs(f_best)+eps)*norm(ub-lb);
                sigma_S=max(sigma_S,1e-6);
                x_pert=max(lb,min(ub,all_x(idx,:)+sigma_S*randn(1,d)));
                [f_pert,cv_pert]=safe_eval(x_pert);
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
                        if row<=size(all_x,1)
                            pop(k).grade{g}(j,:)=all_x(row,:);
                            pop(k).grade_f{g}(j)=all_f(row);
                            pop(k).grade_cv{g}(j)=all_cv(row);
                        end
                    end
                end
            end
        end

        %% Gumii Gayo
        if flags.use_gumii&&mod(t,T_B)==floor(T_B/2)
            if D_t<D_min,     delta_alpha=min(delta_alpha*1.2,2.0);
            elseif D_t>D_max, delta_alpha=delta_alpha*0.85; end
        end

        %% [P1] Problem-type aware Butta trigger
        enough_time = (t-last_butta_t)>=butta_min_interval;
        scheduled   = flags.use_butta && mod(t,T_B)==0;
        late_skip   = scheduled && (D_t>0.5*D_max) && (t-last_butta_t)<T_B*1.5;

        if strcmp(flags.problem_type,'static')
            % Static: Butta only in first 30% OR on diversity collapse
            run_butta = flags.use_butta && enough_time && ~late_skip && ...
                        ((scheduled && t<=0.3*T_max) || (D_t<D_min));
        else
            % Dynamic (MPB): full adaptive Butta throughout
            run_butta = (scheduled||( flags.use_butta&&enough_time&&D_t<D_min)) && ~late_skip;
        end

        if run_butta
            goa_data.butta_iters(end+1) = t;  % record Butta trigger
            [Y_archive,Y_fitness,Y_cv]=yuba_insert_qd(Y_archive,Y_fitness,Y_cv,...
                x_best,f_best,cv_best,NY,sigma_Y);
            [pop,~,~,~,x_best,f_best,cv_best]=...
                GOA_butta_deb(pop,Y_archive,Y_fitness,Y_cv,...
                              x_best,f_best,cv_best,lb,ub,K,NY,safe_eval);
            % [K3] Immediate G1 eval via DE/rand/1 from Yuba (active Gogessas only)
            n_arc=size(Y_archive,1);
            for k=1:K_current
                for j=1:grade_sizes(1)
                    if n_arc>=3
                        r_idx=randperm(n_arc,3);
                        F_g1=M_F(randi(H_mem))+0.1*tan(pi*(rand-0.5));
                        while F_g1<=0
                            F_g1=M_F(randi(H_mem))+0.1*tan(pi*(rand-0.5));
                        end
                        F_g1=min(1,F_g1);
                        x_new=Y_archive(r_idx(1),:)+F_g1*(Y_archive(r_idx(2),:)-Y_archive(r_idx(3),:));
                        if rand<0.4, x_new=x_new+0.3*(x_best-x_new); end
                    else
                        x_new=lb+rand(1,d).*(ub-lb);
                    end
                    x_new=max(lb,min(ub,x_new));
                    [f_new,cv_new]=safe_eval(x_new);
                    pop(k).grade{1}(j,:)=x_new;
                    pop(k).grade_f{1}(j)=f_new;
                    pop(k).grade_cv{1}(j)=cv_new;
                    if deb_compare(f_new,cv_new,f_best,cv_best)
                        f_best=f_new; cv_best=cv_new; x_best=x_new;
                    end
                end
            end
            last_butta_t=t;
        end

        %% Update global best (all K)
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
        convergence(t)=f_best;

        %% Record tracking data for figure generation
        goa_data.diversity_curve(t)   = D_t;
        goa_data.avg_fitness_curve(t) = mean(all_f(~isinf(all_f)));
        goa_data.explore_rate(t)      = min(1, D_t/(D_max+eps));
        goa_data.exploit_rate(t)      = 1 - goa_data.explore_rate(t);
        goa_data.xbest_history(t,:)   = x_best;
        goa_data.K_history(t)         = K_current;
    end  % main loop
    pop_final=pop;
end  % GOA_fixed

%% ========================================================================
function [Y_arc,Y_fit,Y_cv_out]=yuba_insert_qd(Y_arc,Y_fit,Y_cv_out,x_new,f_new,cv_new,NY,sigma_Y)
    if isempty(Y_arc), Y_arc=x_new;Y_fit=f_new;Y_cv_out=cv_new;return;end
    n_arc=size(Y_arc,1);
    dists=vecnorm(Y_arc-x_new,2,2); is_novel=all(dists>sigma_Y);
    [~,worst_idx]=max(Y_fit+1e12*(Y_cv_out>1e-10));
    is_better=deb_compare(f_new,cv_new,Y_fit(worst_idx),Y_cv_out(worst_idx));
    if ~is_novel&&~is_better,return;end
    if n_arc<NY
        Y_arc=[Y_arc;x_new];Y_fit=[Y_fit;f_new];Y_cv_out=[Y_cv_out;cv_new];
    else
        redundant=false(n_arc,1);
        for i=1:n_arc
            od=vecnorm(Y_arc-Y_arc(i,:),2,2);od(i)=inf;
            if min(od)<sigma_Y,redundant(i)=true;end
        end
        if any(redundant)
            ri=find(redundant);[~,lw]=max(Y_fit(ri)+1e12*(Y_cv_out(ri)>1e-10));
            replace_idx=ri(lw);
        else,replace_idx=worst_idx;end
        if is_better||is_novel
            Y_arc(replace_idx,:)=x_new;Y_fit(replace_idx)=f_new;Y_cv_out(replace_idx)=cv_new;
        end
    end
    n=size(Y_arc,1);order=zeros(n,1);remaining=1:n;
    for i=1:n
        best_r=remaining(1);
        for ri=2:numel(remaining)
            if deb_compare(Y_fit(remaining(ri)),Y_cv_out(remaining(ri)),Y_fit(best_r),Y_cv_out(best_r))
                best_r=remaining(ri);end
        end
        order(i)=best_r;remaining=remaining(remaining~=best_r);
    end
    Y_arc=Y_arc(order,:);Y_fit=Y_fit(order);Y_cv_out=Y_cv_out(order);
end

function [x_gb,f_gb]=gogessa_best_local(pop,k)
    f_gb=inf;cv_gb=inf;x_gb=pop(k).grade{6}(1,:);
    for j=1:size(pop(k).grade{6},1)
        if deb_compare(pop(k).grade_f{6}(j),pop(k).grade_cv{6}(j),f_gb,cv_gb)
            f_gb=pop(k).grade_f{6}(j);cv_gb=pop(k).grade_cv{6}(j);x_gb=pop(k).grade{6}(j,:);
        end
    end
end
