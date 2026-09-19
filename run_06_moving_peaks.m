%% run_06_moving_peaks.m
%  FULL Moving Peaks Benchmark — Comprehensive Dynamic Optimization Study
%  --------------------------------------------------
%  Merges standard MPB + dimensional scaling into one script.
%  16 algorithms (13 standard + DynDE, mQSO, CPSO dynamic specialists)
%  --------------------------------------------------
%  11 Scenarios:
%    Standard MPB (D=5):
%      S1: 5 peaks,  s=1.0, freq=3000  (low complexity)
%      S2: 10 peaks, s=1.0, freq=3000  (moderate)
%      S3: 20 peaks, s=3.0, freq=3000  (high complexity)
%      S4: 10 peaks, s=1.0, freq=1500  (fast changes)
%      S5: 10 peaks, s=1.0, freq=7500  (slow changes)
%    Dimensional scaling (10 peaks, s=1.0, freq=3000):
%      S6: D=5   S7: D=10   S8: D=20   S9: D=30
%    Stress (D=10):
%      S10: 10p, s=5.0, freq=3000   (high severity)
%      S11: 10p, s=1.0, freq=1000   (very fast changes)
%  --------------------------------------------------
%  Metrics: Offline Error, Tracking Error, Recovery Curves
%  N=50 | T_max=1000 | 30 runs
%  Outputs: C:\GOA PROJECT\Results\Moving Peaks\
% -----------------------------------------------------------------------
clear; clc; close all;

code_dir    = 'C:\GOA PROJECT\GOA_MATLAB_Code';
results_dir = 'C:\GOA PROJECT\Results\Moving Peaks';
addpath(code_dir); cd(code_dir); clear functions
if ~exist(results_dir,'dir'), mkdir(results_dir); end
diary(fullfile(results_dir,'Console_Output.txt')); diary on;

N=50; T_max=1000; T_B=25; NY=10; n_runs=30;

% Dynamic mode: full Butta throughout (NOT static)
flags.use_butta=true; flags.use_yuba=true; flags.use_topology=true;
flags.use_siinqee=true; flags.use_gumii=true; flags.use_incremental=true;
flags.use_g5shadow=true; flags.beta_val=0.1;
flags.problem_type='dynamic';   % v3.2: full Butta for dynamic problems

%% ===== ALGORITHMS: 16 total (13 standard + 3 dynamic specialists) =====
algo_names = {'GOA','PSO','GWO','WOA','DE','RIME','MVO','TLBO',...
              'SO','SHO','SCSO','SLO','CCO',...
              'DynDE','mQSO','CPSO'};
n_algos = numel(algo_names);

colors_rgb = [0.00 0.40 0.80;  0.80 0.20 0.20;  0.10 0.60 0.10;
              0.60 0.10 0.60;  0.90 0.50 0.00;  0.10 0.70 0.70;
              0.40 0.40 0.40;  0.85 0.33 0.10;
              0.50 0.00 0.50;  0.00 0.60 0.50;  0.70 0.70 0.00;
              0.80 0.00 0.00;  0.00 0.00 0.80;
              0.90 0.20 0.60;  0.20 0.40 0.80;  0.00 0.70 0.00];

%% ===== 11 SCENARIOS =====
scenarios = struct();

% --- Standard MPB (D=5) ---
scenarios(1).name='S1: D=5, 5p, s=1.0 (low)';
scenarios(1).D=5;  scenarios(1).n_peaks=5;  scenarios(1).shift=1.0; scenarios(1).freq=3000;

scenarios(2).name='S2: D=5, 10p, s=1.0 (moderate)';
scenarios(2).D=5;  scenarios(2).n_peaks=10; scenarios(2).shift=1.0; scenarios(2).freq=3000;

scenarios(3).name='S3: D=5, 20p, s=3.0 (complex)';
scenarios(3).D=5;  scenarios(3).n_peaks=20; scenarios(3).shift=3.0; scenarios(3).freq=3000;

scenarios(4).name='S4: D=5, 10p, freq=1500 (fast)';
scenarios(4).D=5;  scenarios(4).n_peaks=10; scenarios(4).shift=1.0; scenarios(4).freq=1500;

scenarios(5).name='S5: D=5, 10p, freq=7500 (slow)';
scenarios(5).D=5;  scenarios(5).n_peaks=10; scenarios(5).shift=1.0; scenarios(5).freq=7500;

% --- Dimensional scaling (10 peaks, s=1.0) ---
scenarios(6).name='S6: D=5, 10p, s=1.0';
scenarios(6).D=5;  scenarios(6).n_peaks=10; scenarios(6).shift=1.0; scenarios(6).freq=3000;

scenarios(7).name='S7: D=10, 10p, s=1.0';
scenarios(7).D=10; scenarios(7).n_peaks=10; scenarios(7).shift=1.0; scenarios(7).freq=3000;

scenarios(8).name='S8: D=20, 10p, s=1.0';
scenarios(8).D=20; scenarios(8).n_peaks=10; scenarios(8).shift=1.0; scenarios(8).freq=3000;

scenarios(9).name='S9: D=30, 10p, s=1.0';
scenarios(9).D=30; scenarios(9).n_peaks=10; scenarios(9).shift=1.0; scenarios(9).freq=3000;

% --- Stress tests (D=10) ---
scenarios(10).name='S10: D=10, 10p, s=5.0 (high sev)';
scenarios(10).D=10; scenarios(10).n_peaks=10; scenarios(10).shift=5.0; scenarios(10).freq=3000;

scenarios(11).name='S11: D=10, 10p, freq=1000 (very fast)';
scenarios(11).D=10; scenarios(11).n_peaks=10; scenarios(11).shift=1.0; scenarios(11).freq=1000;

n_scenarios = numel(scenarios);

%% ===== PRE-ALLOCATE =====
offline_errors  = zeros(n_scenarios, n_algos, n_runs);
tracking_errors = zeros(n_scenarios, n_algos, n_runs);
n_changes_log   = zeros(n_scenarios, n_algos, n_runs);

recovery_window = 200;
recovery_curves = zeros(n_scenarios, n_algos, recovery_window);
recovery_counts = zeros(n_scenarios, n_algos);

%% ========================================================================
%                        MAIN LOOP
%% ========================================================================
t_start = tic;
fprintf('\n%s\n  MOVING PEAKS BENCHMARK — Full Study\n',repmat('=',1,80));
fprintf('  Algorithms: %d (13 standard + DynDE/mQSO/CPSO)\n',n_algos);
fprintf('  Scenarios: %d | Runs: %d | N=%d | T_max=%d\n%s\n\n',...
        n_scenarios,n_runs,N,T_max,repmat('=',1,80));

for si=1:n_scenarios
    sc=scenarios(si);
    fprintf('%s\n  %s | freq=%d FEs\n%s\n',...
            repmat('-',1,80),sc.name,sc.freq,repmat('-',1,80));

    % GOA tracking accumulators for E/E and diversity figures
    goa_div_sc  = zeros(n_scenarios,T_max);
    goa_exp_sc  = zeros(n_scenarios,T_max);
    goa_avg_sc  = zeros(n_scenarios,T_max);
    goa_but_sc  = zeros(n_scenarios,T_max);

    for ai=1:n_algos
        t_algo=tic;
        for run=1:n_runs
            mpb=MovingPeaks(sc.D,sc.n_peaks,sc.freq,sc.shift);
            mpb.reset(run);
            obj_min=@(x) mpb.evaluate_min(x);
            lb_s=mpb.lb; ub_s=mpb.ub;

            [~,~,~,~,gd]=run_algo(algo_names{ai},obj_min,lb_s,ub_s,N,T_max,T_B,NY,flags);

            % Accumulate GOA tracking
            if strcmp(algo_names{ai},'GOA') && ~isempty(gd)
                goa_div_sc(si,:) = goa_div_sc(si,:) + gd.diversity_curve';
                goa_exp_sc(si,:) = goa_exp_sc(si,:) + gd.explore_rate';
                goa_avg_sc(si,:) = goa_avg_sc(si,:) + gd.avg_fitness_curve';
                bv=zeros(1,T_max); bv(gd.butta_iters)=1;
                goa_but_sc(si,:) = goa_but_sc(si,:) + bv;
            end

            true_opt=max(mpb.heights);
            if mpb.best_found>-inf
                mpb.tracking_errors(end+1)=true_opt-mpb.best_found;
            end

            offline_errors(si,ai,run) =mpb.get_offline_error();
            tracking_errors(si,ai,run)=mpb.get_mean_tracking_error();
            n_changes_log(si,ai,run)  =mpb.n_changes;

            % Recovery curves
            n_fe=min(mpb.fe_count,numel(mpb.fe_fitness_log));
            for ci=1:numel(mpb.change_fe_points)
                cp=mpb.change_fe_points(ci);
                if cp+recovery_window>n_fe, continue; end
                true_after=mpb.fe_true_opt_log(cp+1);
                if true_after<=0, continue; end
                for fe=1:recovery_window
                    idx_fe=cp+fe;
                    if idx_fe<=n_fe && true_after>0
                        ratio=mpb.fe_fitness_log(idx_fe)/true_after;
                        recovery_curves(si,ai,fe)=recovery_curves(si,ai,fe)+ratio;
                    end
                end
                recovery_counts(si,ai)=recovery_counts(si,ai)+1;
            end
        end

        if recovery_counts(si,ai)>0
            recovery_curves(si,ai,:)=recovery_curves(si,ai,:)/recovery_counts(si,ai);
        end

        mean_oe=mean(squeeze(offline_errors(si,ai,:)));
        std_oe =std(squeeze(offline_errors(si,ai,:)));
        mean_te=mean(squeeze(tracking_errors(si,ai,:)));
        mean_nc=mean(squeeze(n_changes_log(si,ai,:)));
        fprintf('  %-12s | OE=%.2f+/-%.2f | TE=%.2f | Chg=%.0f | %.1fs\n',...
                algo_names{ai},mean_oe,std_oe,mean_te,mean_nc,toc(t_algo));
    end
    fprintf('\n');
end

elapsed=toc(t_start);
fprintf('\n%s\n  Total: %.1f min\n%s\n',repmat('=',1,80),elapsed/60,repmat('=',1,80));

%% ===== RANKING =====
fprintf('\n%s\n  FRIEDMAN RANK — OFFLINE ERROR\n%s\n',repmat('=',1,80),repmat('=',1,80));

OE_matrix=zeros(n_scenarios,n_algos);
for si=1:n_scenarios
    for ai=1:n_algos
        OE_matrix(si,ai)=mean(squeeze(offline_errors(si,ai,:)));
    end
end
R=zeros(n_scenarios,n_algos);
for si=1:n_scenarios
    [~,ord]=sort(OE_matrix(si,:));
    r_tmp=zeros(1,n_algos);
    for ri=1:n_algos, r_tmp(ord(ri))=ri; end
    R(si,:)=r_tmp;
end
rank_mpb=mean(R,1)';

[~,sr]=sort(rank_mpb);
for ai=1:n_algos
    fprintf('  %2d. %-12s  Rank=%.2f  MeanOE=%.2f\n',...
            ai,algo_names{sr(ai)},rank_mpb(sr(ai)),mean(OE_matrix(:,sr(ai))));
end

% Dimensional scaling analysis (S6-S9)
fprintf('\n%s\n  DIMENSIONAL SCALING (S6-S9)\n%s\n',repmat('=',1,80),repmat('=',1,80));
dim_s=[6,7,8,9]; dim_vals=[5,10,20,30];
fprintf('\n  %-12s  D=5   D=10  D=20  D=30\n','Algorithm');
fprintf('  %s\n',repmat('-',1,50));
for ai=1:n_algos
    fprintf('  %-12s',algo_names{ai});
    for di=1:4, fprintf('  %5.1f',OE_matrix(dim_s(di),ai)); end
    fprintf('\n');
end

%% ===== CSV =====
fid=fopen(fullfile(results_dir,'Table_MPB_Full.csv'),'w');
fprintf(fid,'Algorithm');
for si=1:n_scenarios, fprintf(fid,',%s_OE,%s_TE',scenarios(si).name,scenarios(si).name); end
fprintf(fid,',MeanOE,Rank\n');
for ai=1:n_algos
    fprintf(fid,'%s',algo_names{ai});
    for si=1:n_scenarios
        fprintf(fid,',%.4f,%.4f',...
            mean(squeeze(offline_errors(si,ai,:))),...
            mean(squeeze(tracking_errors(si,ai,:))));
    end
    fprintf(fid,',%.4f,%.2f\n',mean(OE_matrix(:,ai)),rank_mpb(ai));
end
fclose(fid);

%% ===== SAVE — before figures so crash never loses data =====
% Normalise GOA tracking
goa_tracking.diversity   = goa_div_sc / n_runs;
goa_tracking.explore     = goa_exp_sc / n_runs;
goa_tracking.exploit     = 1 - goa_exp_sc/n_runs;
goa_tracking.avg_fitness = goa_avg_sc / n_runs;
goa_tracking.butta_freq  = goa_but_sc / n_runs;
goa_tracking.scenario_names = {scenarios.name};

save(fullfile(results_dir,'moving_peaks_full_results.mat'),...
     'offline_errors','tracking_errors','n_changes_log',...
     'recovery_curves','recovery_counts','OE_matrix','rank_mpb',...
     'algo_names','scenarios','n_runs','T_max','N','elapsed',...
     'goa_tracking');
fprintf('  Data saved to: %s\n',results_dir);

%% ===== FIGURES =====
fprintf('  Generating figures...\n');

% Friedman rank
fig=figure('Visible','off','Position',[100 100 1000 450]);
b=bar(1:n_algos,rank_mpb,0.7); b.FaceColor='flat';
for ai=1:n_algos, b.CData(ai,:)=colors_rgb(ai,:); end
hold on;
for ai=1:n_algos
    text(ai,rank_mpb(ai)+0.15,sprintf('%.2f',rank_mpb(ai)),...
        'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
end
set(gca,'XTick',1:n_algos,'XTickLabel',algo_names,'XTickLabelRotation',45,'FontSize',9);
ylabel('Mean Rank','FontSize',12);
title('Moving Peaks — Friedman Rank (11 scenarios)','FontSize',13);
ylim([0 n_algos+1]); grid on; box on; hold off;
saveas(fig,fullfile(results_dir,'Fig_Friedman_MPB.png')); close(fig);

% Dimensional scaling
fig=figure('Visible','off','Position',[100 100 900 500]);
hold on;
for ai=1:n_algos
    oe_vals_dim = OE_matrix(dim_s, ai);  % 4×1 vector for this algorithm
    lw=1.2; if ai==1, lw=3; end; if ai>=14, lw=2; end
    plot(dim_vals, oe_vals_dim', '-o','Color',colors_rgb(ai,:),'LineWidth',lw,'MarkerSize',6);
end
xlabel('Dimension D','FontSize',13); ylabel('Offline Error','FontSize',13);
title('MPB Dimensional Scaling','FontSize',14);
legend(algo_names,'Location','eastoutside','FontSize',7);
set(gca,'XTick',dim_vals,'FontSize',11);
grid on; box on; hold off;
saveas(fig,fullfile(results_dir,'Fig_Dimensional_Scaling.png')); close(fig);

% Recovery curve (S2: D=5, 10 peaks)
si_rc=2;
fig=figure('Visible','off','Position',[100 100 800 450]);
hold on;
for ai=1:n_algos
    rc=squeeze(recovery_curves(si_rc,ai,:));
    if any(rc>0)
        lw=1.0; if ai==1, lw=2.5; end; if ai>=14, lw=2; end
        plot(1:recovery_window,rc,'-','Color',colors_rgb(ai,:),'LineWidth',lw);
    end
end
yline(1.0,'--k','LineWidth',1);
xlabel('FEs After Change','FontSize',12); ylabel('Best Found / True Optimum','FontSize',12);
title(sprintf('Recovery — %s',scenarios(si_rc).name),'FontSize',13);
legend(algo_names,'Location','southeast','FontSize',7);
ylim([0 1.1]); xlim([1 recovery_window]); grid on; box on; hold off;
saveas(fig,fullfile(results_dir,'Fig_Recovery_S2.png')); close(fig);

% OE heatmap
fig=figure('Visible','off','Position',[100 100 1000 600]);
imagesc(OE_matrix');
colormap(flipud(hot)); colorbar;
set(gca,'XTick',1:n_scenarios,...
    'XTickLabel',arrayfun(@(i)sprintf('S%d',i),1:n_scenarios,'UniformOutput',false),...
    'YTick',1:n_algos,'YTickLabel',algo_names,'FontSize',9);
xlabel('Scenario','FontSize',12); ylabel('Algorithm','FontSize',12);
title('Offline Error Heatmap','FontSize',13);
for si=1:n_scenarios
    for ai=1:n_algos
        v=OE_matrix(si,ai);
        if v<median(OE_matrix(:)), tc='w'; else, tc='k'; end
        text(si,ai,sprintf('%.0f',v),'HorizontalAlignment','center',...
            'Color',tc,'FontSize',7,'FontWeight','bold');
    end
end
saveas(fig,fullfile(results_dir,'Fig_OE_Heatmap.png')); close(fig);

%% ===== SAVE =====
fprintf('\n  Moving Peaks complete.\n\n');
diary off;

%% ========================================================================
function [x_best,f_best,convergence,pop_out,goa_data]=run_algo(name,obj,lb,ub,N,T_max,T_B,NY,flags)
    goa_data=[]; pop_out=[];
    switch name
        case 'GOA',  [x_best,f_best,convergence,~,goa_tracking_out]=GOA_fixed(obj,lb,ub,N,T_max,T_B,NY,flags);
        case 'PSO',    [x_best,f_best,convergence]=PSO(obj,lb,ub,N,T_max);
        case 'GWO',    [x_best,f_best,convergence]=GWO(obj,lb,ub,N,T_max);
        case 'WOA',    [x_best,f_best,convergence]=WOA(obj,lb,ub,N,T_max);
        case 'DE',     [x_best,f_best,convergence]=DE(obj,lb,ub,N,T_max);
        case 'RIME',   [x_best,f_best,convergence]=RIME(obj,lb,ub,N,T_max);
        case 'MVO',    [x_best,f_best,convergence]=MVO(obj,lb,ub,N,T_max);
        case 'TLBO',   [x_best,f_best,convergence]=TLBO(obj,lb,ub,N,T_max);
        case 'SO',     [x_best,f_best,convergence]=SO(obj,lb,ub,N,T_max);
        case 'SHO',    [x_best,f_best,convergence]=SHO(obj,lb,ub,N,T_max);
        case 'SCSO',   [x_best,f_best,convergence]=SCSO(obj,lb,ub,N,T_max);
        case 'SLO',    [x_best,f_best,convergence]=SLO(obj,lb,ub,N,T_max);
        case 'CCO',    [x_best,f_best,convergence]=CCO(obj,lb,ub,N,T_max);
        case 'DynDE',  [x_best,f_best,convergence]=DynDE(obj,lb,ub,N,T_max);
        case 'mQSO',   [x_best,f_best,convergence]=mQSO(obj,lb,ub,N,T_max);
        case 'CPSO',   [x_best,f_best,convergence]=CPSO(obj,lb,ub,N,T_max);
        otherwise, error('Unknown: %s',name);
    end
end
