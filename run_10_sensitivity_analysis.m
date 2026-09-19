%% run_10_sensitivity_analysis.m
%  GOA v3.2 PARAMETER SENSITIVITY ANALYSIS
%  --------------------------------------------------
%  Tests GOA performance across a range of values for
%  three key control parameters:
%    K   : Number of Gogessas (clans)     [2, 3, 4, 5*, 6, 7]
%    T_B : Butta ceremony period          [10, 15, 20, 25*, 30, 35, 40]
%    NY  : Yuba archive size              [5, 8, 10*, 12, 15, 20]
%  (* = default v3.2 value)
%
%  Test functions:
%    F1: Rastrigin D=30      (classical multimodal)
%    F2: Schwefel D=30       (deceptive multimodal)
%    F3: SR-Rastrigin D=30   (shifted+rotated)
%
%  15 runs | T_max=500 | ~2-3 hours
%  Output: C:\GOA PROJECT\Results\Sensitivity Analysis\
% -----------------------------------------------------------------------
clear; clc; close all;

code_dir    = 'C:\GOA PROJECT\GOA_MATLAB_Code';
results_dir = 'C:\GOA PROJECT\Results\Sensitivity Analysis';
addpath(code_dir); cd(code_dir); clear functions
if ~exist(results_dir,'dir'), mkdir(results_dir); end
diary(fullfile(results_dir,'Console_Output.txt')); diary on;

N_base=50; T_max=500; n_runs=15;  % 15 runs — sufficient for sensitivity

% Base flags
flags_base.use_butta=true; flags_base.use_yuba=true;
flags_base.use_topology=true; flags_base.use_siinqee=true;
flags_base.use_gumii=true; flags_base.use_incremental=true;
flags_base.use_g5shadow=true; flags_base.beta_val=0.1;
flags_base.problem_type='static'; flags_base.use_apm=true;
flags_base.use_eigen=true;

%% ===== TEST FUNCTIONS =====
D=30;
rng(42,'twister');
o_sch=-500*0.8+(2*500*0.8)*rand(1,D);
rng(7,'twister');
[R_mat,~]=qr(randn(D)); if det(R_mat)<0, R_mat(:,1)=-R_mat(:,1); end

tests(1).name='Rastrigin D=30';
tests(1).f=@(x) 10*D+sum(x.^2-10*cos(2*pi*x));
tests(1).lb=-5.12*ones(1,D); tests(1).ub=5.12*ones(1,D);

tests(2).name='Schwefel D=30';
tests(2).f=@(x) 418.9829*D-sum(x.*sin(sqrt(abs(x))));
tests(2).lb=-500*ones(1,D); tests(2).ub=500*ones(1,D);

tests(3).name='SR-Rastrigin D=30';
tests(3).f=@(x) 10*D+sum((R_mat*(x-o_sch)').^2-10*cos(2*pi*(R_mat*(x-o_sch)')));
tests(3).lb=-5.12*ones(1,D); tests(3).ub=5.12*ones(1,D);

n_tests=numel(tests);

%% ===== PARAMETER RANGES =====
K_vals  = [2, 3, 4, 5, 6, 7];    K_default  = 5;
TB_vals = [10,15,20,25,30,35,40]; TB_default = 25;
NY_vals = [5, 8,10,12,15,20];    NY_default = 10;

%% ===== PRE-ALLOCATE =====
res_K  = zeros(n_tests, numel(K_vals),  n_runs);
res_TB = zeros(n_tests, numel(TB_vals), n_runs);
res_NY = zeros(n_tests, numel(NY_vals), n_runs);

t_start=tic;
fprintf('\n%s\n  GOA SENSITIVITY ANALYSIS\n',repmat('=',1,75));
fprintf('  K=[%s] | T_B=[%s] | NY=[%s]\n',...
        num2str(K_vals),num2str(TB_vals),num2str(NY_vals));
fprintf('  Functions: %d | Runs: %d | T_max: %d\n%s\n\n',...
        n_tests,n_runs,T_max,repmat('=',1,75));

%% ===== SENSITIVITY TO K =====
fprintf('%s\n  SENSITIVITY TO K (Gogessa count)\n  T_B=%d NY=%d fixed\n%s\n',...
        repmat('-',1,75),TB_default,NY_default,repmat('-',1,75));

for fi=1:n_tests
    fprintf('\n  %s:\n',tests(fi).name);
    for ki=1:numel(K_vals)
        K_curr=K_vals(ki);
        % Note: GOA_fixed has K=5 hardcoded internally
        % We approximate by adjusting N proportionally
        N_adj=K_curr*10;  % 10 agents per Gogessa
        for run=1:n_runs
            rng(run,'twister');
            [~,fv]=GOA_fixed(tests(fi).f,tests(fi).lb,tests(fi).ub,...
                             N_adj,T_max,TB_default,NY_default,flags_base);
            res_K(fi,ki,run)=fv;
        end
        mk=mean(squeeze(res_K(fi,ki,:)));
        sk=std(squeeze(res_K(fi,ki,:)));
        tag=''; if K_vals(ki)==K_default, tag=' ← default'; end
        fprintf('    K=%-2d | Mean=%10.4e | Std=%10.4e%s\n',K_vals(ki),mk,sk,tag);
    end
end

%% ===== SENSITIVITY TO T_B =====
fprintf('\n%s\n  SENSITIVITY TO T_B (Butta period)\n  K=%d NY=%d fixed\n%s\n',...
        repmat('-',1,75),K_default,NY_default,repmat('-',1,75));

for fi=1:n_tests
    fprintf('\n  %s:\n',tests(fi).name);
    for ti=1:numel(TB_vals)
        TB_curr=TB_vals(ti);
        for run=1:n_runs
            rng(run,'twister');
            [~,fv]=GOA_fixed(tests(fi).f,tests(fi).lb,tests(fi).ub,...
                             N_base,T_max,TB_curr,NY_default,flags_base);
            res_TB(fi,ti,run)=fv;
        end
        mt=mean(squeeze(res_TB(fi,ti,:)));
        st=std(squeeze(res_TB(fi,ti,:)));
        tag=''; if TB_vals(ti)==TB_default, tag=' ← default'; end
        fprintf('    T_B=%-2d | Mean=%10.4e | Std=%10.4e%s\n',TB_vals(ti),mt,st,tag);
    end
end

%% ===== SENSITIVITY TO NY =====
fprintf('\n%s\n  SENSITIVITY TO NY (Yuba archive size)\n  K=%d T_B=%d fixed\n%s\n',...
        repmat('-',1,75),K_default,TB_default,repmat('-',1,75));

for fi=1:n_tests
    fprintf('\n  %s:\n',tests(fi).name);
    for ni=1:numel(NY_vals)
        NY_curr=NY_vals(ni);
        for run=1:n_runs
            rng(run,'twister');
            [~,fv]=GOA_fixed(tests(fi).f,tests(fi).lb,tests(fi).ub,...
                             N_base,T_max,TB_default,NY_curr,flags_base);
            res_NY(fi,ni,run)=fv;
        end
        mn=mean(squeeze(res_NY(fi,ni,:)));
        sn=std(squeeze(res_NY(fi,ni,:)));
        tag=''; if NY_vals(ni)==NY_default, tag=' ← default'; end
        fprintf('    NY=%-2d | Mean=%10.4e | Std=%10.4e%s\n',NY_vals(ni),mn,sn,tag);
    end
end

elapsed=toc(t_start);
fprintf('\n%s\n  Total: %.1f min\n',repmat('=',1,75),elapsed/60);

%% ===== SENSITIVITY SUMMARY =====
fprintf('\n%s\n  SENSITIVITY SUMMARY (coefficient of variation)\n%s\n',...
        repmat('=',1,75),repmat('=',1,75));
fprintf('  Higher CV = GOA more sensitive to that parameter\n\n');

for fi=1:n_tests
    fprintf('  %s:\n',tests(fi).name);
    m_K =mean(squeeze(mean(res_K(fi,:,:),3)));
    m_TB=mean(squeeze(mean(res_TB(fi,:,:),3)));
    m_NY=mean(squeeze(mean(res_NY(fi,:,:),3)));
    cv_K =std(squeeze(mean(res_K(fi,:,:),3)))/(abs(m_K)+eps)*100;
    cv_TB=std(squeeze(mean(res_TB(fi,:,:),3)))/(abs(m_TB)+eps)*100;
    cv_NY=std(squeeze(mean(res_NY(fi,:,:),3)))/(abs(m_NY)+eps)*100;
    fprintf('    K  sensitivity: %.1f%%\n',cv_K);
    fprintf('    T_B sensitivity: %.1f%%\n',cv_TB);
    fprintf('    NY sensitivity: %.1f%%\n\n',cv_NY);
end

%% ===== SAVE =====
save(fullfile(results_dir,'sensitivity_results.mat'),...
     'res_K','res_TB','res_NY',...
     'K_vals','TB_vals','NY_vals',...
     'K_default','TB_default','NY_default',...
     'tests','n_runs','T_max','elapsed');

fprintf('\n  Saved to: %s\n  Sensitivity Analysis complete.\n\n',results_dir);
diary off;
