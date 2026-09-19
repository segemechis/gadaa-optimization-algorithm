%% run_02_CEC2022.m
%  CEC-2022 Benchmark — All 12 Functions (F1-F12)
%  --------------------------------------------------
%  Official dimensions: D=10, 20 (D=2 excluded — Hybrid/Composition undefined)
%  F1-F5:  Simple functions (unimodal/multimodal)
%  F6-F8:  Hybrid functions
%  F9-F12: Composition functions
%  --------------------------------------------------
%  13 algorithms | N=50 | T_max=500 | 30 runs
%  Error = f(x) - f* where f* = func_num * 1000
%  Input: COLUMN vector x (D×1) — CEC-2022 requirement
%  --------------------------------------------------
%  Output: C:\GOA PROJECT\Results\CEC2022 Benchmark\
% -----------------------------------------------------------------------
clear; clc; close all;

code_dir    = 'C:\GOA PROJECT\GOA_MATLAB_Code';
cec22_dir   = fullfile(code_dir,'CEC2022');
results_dir = 'C:\GOA PROJECT\Results\CEC2022 Benchmark';

addpath(code_dir);
addpath(cec22_dir);
cd(cec22_dir);   % MUST cd here — input_data/ read relative to cwd
clear functions

if ~exist(results_dir,'dir'), mkdir(results_dir); end
diary(fullfile(results_dir,'Console_Output.txt')); diary on;

N=50; T_max=500; T_B=25; NY=10; n_runs=30;

flags.use_butta=true; flags.use_yuba=true; flags.use_topology=true;
flags.use_siinqee=true; flags.use_gumii=true; flags.use_incremental=true;
flags.use_g5shadow=true; flags.beta_val=0.1;
flags.problem_type='static';   % v3.2: limit Butta to early phase

algo_names={'GOA','PSO','GWO','WOA','DE','RIME','MVO','TLBO',...
            'SO','SHO','SCSO','SLO','CCO'};
n_algos=numel(algo_names);

colors_rgb=[0.00 0.40 0.80;0.80 0.20 0.20;0.10 0.60 0.10;
            0.60 0.10 0.60;0.90 0.50 0.00;0.10 0.70 0.70;
            0.40 0.40 0.40;0.85 0.33 0.10;0.50 0.00 0.50;
            0.00 0.60 0.50;0.70 0.70 0.00;0.80 0.00 0.00;0.00 0.00 0.80];

%% ===== CEC-2022 FUNCTION SET =====
func_nums  = 1:12;
n_funcs    = 12;
func_types = {'Simple','Simple','Simple','Simple','Simple',...   % F1-F5
              'Hybrid','Hybrid','Hybrid',...                     % F6-F8
              'Composition','Composition','Composition','Composition'}; % F9-F12
func_labels= {'F1:Shifted Bent Cigar','F2:Shifted Schwefel','F3:Shifted Lunacek',...
              'F4:Shifted Rosenbrock','F5:Shifted Rastrigin',...
              'F6:Hybrid1 (N=3)','F7:Hybrid2 (N=4)','F8:Hybrid3 (N=4)',...
              'F9:Composition1 (N=3)','F10:Composition2 (N=3)',...
              'F11:Composition3 (N=4)','F12:Composition4 (N=4)'};

% CEC-2022 supported dimensions: D=10 and D=20
% NOTE: D=2 excluded — Hybrid (F6-F8) and Composition (F9-F12)
% functions are NOT defined for D=2 (no shuffle_data files).
% D=10 and D=20 are the standard competition dimensions.
dims   = [10, 20];
n_dims = numel(dims);

% Correct bias values from CEC-2022 technical report (Kumar et al., 2022)
% These are the actual global optimum values f(x*) for each function
% Verified from results: F1 reaches ~300, F2 ~400, F3 ~600, etc.
f_bias = [300, 400, 600, 800, 900, ...    % F1-F5  Simple
          1800, 2000, 2200, ...            % F6-F8  Hybrid
          2300, 2400, 2600, 2700];         % F9-F12 Composition

%% ===== PRE-ALLOCATE =====
results = zeros(n_funcs, n_dims, n_algos, n_runs);   % error = f - f*
conv    = zeros(n_funcs, n_dims, n_algos, T_max, n_runs);

%% ========================================================================
t_start=tic;
fprintf('\n%s\n  CEC-2022 BENCHMARK — All 12 Functions\n',repmat('=',1,80));
fprintf('  Dimensions: [2 10 20] | Algorithms: %d | Runs: %d\n',n_algos,n_runs);
fprintf('  NOTE: input is column vector (D×1), D=2/10/20 only\n%s\n\n',...
        repmat('=',1,80));

for di=1:n_dims
    D=dims(di);
    lb=-100*ones(1,D); ub=100*ones(1,D);

    fprintf('%s\n  D=%d\n%s\n',repmat('-',1,80),D,repmat('-',1,80));

    for fi=1:n_funcs
        fnum  = func_nums(fi);
        f_star= f_bias(fi);

        % Wrapper: optimizer gives row vector → transpose to column
        obj = @(x) cec22_test_func(x(:), fnum);

        fprintf('\n  [%d/%d] %s | D=%d | f*=%d\n',...
                fi,n_funcs,func_labels{fi},D,f_star);

        for ai=1:n_algos
            bv=zeros(n_runs,1); cv=zeros(T_max,n_runs);
            for run=1:n_runs
                rng(run,'twister');
                [~,fv,c]=run_algo(algo_names{ai},obj,lb,ub,N,T_max,T_B,NY,flags);
                bv(run)=fv-f_star;
                cv(:,run)=c-f_star;
            end
            results(fi,di,ai,:)=bv;
            conv(fi,di,ai,:,:)=cv;
            fprintf('    %-8s | Mean=%10.4e | Std=%10.4e | Best=%10.4e\n',...
                    algo_names{ai},mean(bv),std(bv),min(bv));
        end
    end
end

elapsed=toc(t_start);
fprintf('\n%s\n  Total: %.1f min\n%s\n',repmat('=',1,80),elapsed/60,repmat('=',1,80));

%% ===== FRIEDMAN RANK =====
fprintf('\n%s\n  FRIEDMAN RANK ANALYSIS\n%s\n',repmat('=',1,80),repmat('=',1,80));

rank_per_dim=zeros(n_algos,n_dims);
for di=1:n_dims
    M=zeros(n_funcs,n_algos);
    for fi=1:n_funcs
        for ai=1:n_algos
            M(fi,ai)=mean(squeeze(results(fi,di,ai,:)));
        end
    end
    R=zeros(n_funcs,n_algos);
    for fi=1:n_funcs
        [~,ord]=sort(M(fi,:));
        r_tmp=zeros(1,n_algos);
        for ri=1:n_algos, r_tmp(ord(ri))=ri; end
        R(fi,:)=r_tmp;
    end
    rank_per_dim(:,di)=mean(R,1)';
    fprintf('\n  D=%d:\n',dims(di));
    [~,sr]=sort(rank_per_dim(:,di));
    for ai=1:n_algos
        fprintf('    %2d. %-8s  Rank=%.2f\n',ai,algo_names{sr(ai)},rank_per_dim(sr(ai),di));
    end
end

rank_combined=mean(rank_per_dim,2);
fprintf('\n  COMBINED:\n');
[~,sr]=sort(rank_combined);
for ai=1:n_algos
    fprintf('    %2d. %-8s  Rank=%.2f\n',ai,algo_names{sr(ai)},rank_combined(sr(ai)));
end

%% ===== TYPE ANALYSIS (D=10) =====
fprintf('\n%s\n  TYPE ANALYSIS — D=10\n%s\n',repmat('=',1,80),repmat('=',1,80));
di10=find(dims==10);
type_names={'Simple','Hybrid','Composition'};
type_idx={1:5, 6:8, 9:12};
M10=zeros(n_funcs,n_algos);
for fi=1:n_funcs
    for ai=1:n_algos
        M10(fi,ai)=mean(squeeze(results(fi,di10,ai,:)));
    end
end
for ti=1:3
    fi_set=type_idx{ti};
    M_t=M10(fi_set,:);
    R_t=zeros(numel(fi_set),n_algos);
    for fi=1:numel(fi_set)
        [~,ord]=sort(M_t(fi,:));
        r_tmp=zeros(1,n_algos);
        for ri=1:n_algos, r_tmp(ord(ri))=ri; end
        R_t(fi,:)=r_tmp;
    end
    rank_t=mean(R_t,1);
    fprintf('\n  %s:\n',type_names{ti});
    [~,sr]=sort(rank_t);
    for ai=1:min(5,n_algos)
        fprintf('    %d. %-8s Rank=%.2f\n',ai,algo_names{sr(ai)},rank_t(sr(ai)));
    end
end

%% ===== FRIEDMAN TEST (D=10) =====
R10=zeros(n_funcs,n_algos);
for fi=1:n_funcs
    [~,ord]=sort(M10(fi,:));
    r_tmp=zeros(1,n_algos);
    for ri=1:n_algos, r_tmp(ord(ri))=ri; end
    R10(fi,:)=r_tmp;
end
R10_mean=mean(R10,1);
k=n_algos; n_obs=n_funcs;
chi2_val=12*n_obs/(k*(k+1))*(sum(R10_mean.^2)-k*(k+1)^2/4);
p_frd=1-chi2cdf(chi2_val,k-1);
fprintf('\n  Friedman test D=10: chi2=%.2f, p=%.6f\n',chi2_val,p_frd);

%% ===== CSV =====
fprintf('\n  Saving CSV...\n');
for di=1:n_dims
    fid=fopen(fullfile(results_dir,sprintf('Table_CEC2022_D%d.csv',dims(di))),'w');
    fprintf(fid,'Function,Type,Metric');
    for ai=1:n_algos, fprintf(fid,',%s',algo_names{ai}); end
    fprintf(fid,'\n');
    for fi=1:n_funcs
        m=zeros(n_algos,1); s=zeros(n_algos,1);
        for ai=1:n_algos
            v=squeeze(results(fi,di,ai,:)); m(ai)=mean(v); s(ai)=std(v);
        end
        fprintf(fid,'%s,%s,Mean',func_labels{fi},func_types{fi});
        for ai=1:n_algos, fprintf(fid,',%.4e',m(ai)); end
        fprintf(fid,'\n,,Std');
        for ai=1:n_algos, fprintf(fid,',%.4e',s(ai)); end
        fprintf(fid,'\n');
    end
    fprintf(fid,'\nFriedman Rank,,');
    for ai=1:n_algos, fprintf(fid,',%.2f',rank_per_dim(ai,di)); end
    fprintf(fid,'\n');
    fclose(fid);
end
fid=fopen(fullfile(results_dir,'Table_Friedman_CEC2022.csv'),'w');
fprintf(fid,'Algorithm');
for di=1:n_dims, fprintf(fid,',D=%d',dims(di)); end
fprintf(fid,',Combined\n');
for ai=1:n_algos
    fprintf(fid,'%s',algo_names{ai});
    for di=1:n_dims, fprintf(fid,',%.2f',rank_per_dim(ai,di)); end
    fprintf(fid,',%.2f\n',rank_combined(ai));
end
fclose(fid);

%% ===== FIGURES =====
fprintf('  Generating figures...\n');

% Combined rank
fig=figure('Visible','off','Position',[100 100 900 450]);
b=bar(1:n_algos,rank_combined,0.7); b.FaceColor='flat';
for ai=1:n_algos, b.CData(ai,:)=colors_rgb(ai,:); end
hold on;
for ai=1:n_algos
    text(ai,rank_combined(ai)+0.15,sprintf('%.2f',rank_combined(ai)),...
        'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
end
set(gca,'XTick',1:n_algos,'XTickLabel',algo_names,'XTickLabelRotation',45,'FontSize',10);
ylabel('Mean Rank','FontSize',12);
title('CEC-2022 Combined Friedman Rank (12 functions)','FontSize',13);
ylim([0 n_algos+1]); grid on; box on; hold off;
saveas(fig,fullfile(results_dir,'Fig_Friedman_Combined.png')); close(fig);

% Rank per dimension
for di=1:n_dims
    fig=figure('Visible','off','Position',[100 100 900 450]);
    b=bar(1:n_algos,rank_per_dim(:,di),0.7); b.FaceColor='flat';
    for ai=1:n_algos, b.CData(ai,:)=colors_rgb(ai,:); end
    hold on;
    for ai=1:n_algos
        text(ai,rank_per_dim(ai,di)+0.15,sprintf('%.2f',rank_per_dim(ai,di)),...
            'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');
    end
    set(gca,'XTick',1:n_algos,'XTickLabel',algo_names,'XTickLabelRotation',45,'FontSize',10);
    ylabel('Mean Rank'); title(sprintf('CEC-2022 Friedman Rank D=%d',dims(di)));
    ylim([0 n_algos+1]); grid on; box on; hold off;
    saveas(fig,fullfile(results_dir,sprintf('Fig_Friedman_D%d.png',dims(di)))); close(fig);
end

% Convergence D=10
di10=find(dims==10);
for fi=1:n_funcs
    fig=figure('Visible','off','Position',[100 100 800 450]);
    hold on;
    for ai=1:n_algos
        mc=mean(squeeze(conv(fi,di10,ai,:,:)),2);
        semilogy(1:T_max,max(mc,1e-10),'-','Color',colors_rgb(ai,:),'LineWidth',1.2);
    end
    xlabel('Iteration'); ylabel('Error (f-f*)');
    title(sprintf('%s D=10',func_labels{fi}));
    legend(algo_names,'Location','northeast','FontSize',7);
    set(gca,'YScale','log'); grid on; box on; hold off;
    saveas(fig,fullfile(results_dir,sprintf('Fig_Conv_F%d_D10.png',fnum)));
    close(fig);
end

% Scalability
fig=figure('Visible','off','Position',[100 100 800 450]);
hold on;
for ai=1:n_algos
    plot(dims,rank_per_dim(ai,:),'-o','Color',colors_rgb(ai,:),'LineWidth',1.5,'MarkerSize',6);
end
xlabel('Dimension D'); ylabel('Mean Rank');
title('CEC-2022 Rank vs Dimension');
legend(algo_names,'Location','eastoutside','FontSize',8);
set(gca,'XTick',dims); ylim([0 n_algos+1]); grid on; box on; hold off;
saveas(fig,fullfile(results_dir,'Fig_Scalability.png')); close(fig);

%% ===== SAVE =====
save(fullfile(results_dir,'CEC2022_results.mat'),...
     'results','conv','algo_names','func_nums','func_labels','func_types',...
     'dims','n_runs','T_max','rank_per_dim','rank_combined','chi2_val','p_frd','elapsed');

fprintf('\n  All saved to: %s\n  CEC-2022 complete.\n\n',results_dir);
diary off;

%% ========================================================================
function [x_best,f_best,convergence]=run_algo(name,obj,lb,ub,N,T_max,T_B,NY,flags)
    switch name
        case 'GOA',  [x_best,f_best,convergence,~]=GOA_fixed(obj,lb,ub,N,T_max,T_B,NY,flags);
        case 'PSO',  [x_best,f_best,convergence]=PSO(obj,lb,ub,N,T_max);
        case 'GWO',  [x_best,f_best,convergence]=GWO(obj,lb,ub,N,T_max);
        case 'WOA',  [x_best,f_best,convergence]=WOA(obj,lb,ub,N,T_max);
        case 'DE',   [x_best,f_best,convergence]=DE(obj,lb,ub,N,T_max);
        case 'RIME', [x_best,f_best,convergence]=RIME(obj,lb,ub,N,T_max);
        case 'MVO',  [x_best,f_best,convergence]=MVO(obj,lb,ub,N,T_max);
        case 'TLBO', [x_best,f_best,convergence]=TLBO(obj,lb,ub,N,T_max);
        case 'SO',   [x_best,f_best,convergence]=SO(obj,lb,ub,N,T_max);
        case 'SHO',  [x_best,f_best,convergence]=SHO(obj,lb,ub,N,T_max);
        case 'SCSO', [x_best,f_best,convergence]=SCSO(obj,lb,ub,N,T_max);
        case 'SLO',  [x_best,f_best,convergence]=SLO(obj,lb,ub,N,T_max);
        case 'CCO',  [x_best,f_best,convergence]=CCO(obj,lb,ub,N,T_max);
        otherwise,   error('Unknown: %s',name);
    end
end
