%% run_08b_CEC2017_part2.m
%  CEC-2017 Benchmark — Part 2 of 3
%  F11-F20 (10 Hybrid Functions)
%  D=10,30,50 | 13 algorithms | N=50 | T_max=500 | 30 runs
%  Output: C:\GOA PROJECT\Results\CEC2017 Benchmark\part2\
% -----------------------------------------------------------------------
clear; clc; close all;

code_dir    = 'C:\GOA PROJECT\GOA_MATLAB_Code';
results_dir = 'C:\GOA PROJECT\Results\CEC2017 Benchmark\part2';
addpath(code_dir); cd(code_dir); clear functions
if ~exist(results_dir,'dir'), mkdir(results_dir); end
diary(fullfile(results_dir,'Console_Output_Part2.txt')); diary on;

N=50; T_max=500; T_B=25; NY=10; n_runs=30;

flags.use_butta=true; flags.use_yuba=true; flags.use_topology=true;
flags.use_siinqee=true; flags.use_gumii=true; flags.use_incremental=true;
flags.use_g5shadow=true; flags.beta_val=0.1;
flags.problem_type='static';

algo_names={'GOA','PSO','GWO','WOA','DE','RIME','MVO','TLBO',...
            'SO','SHO','SCSO','SLO','CCO'};
n_algos=numel(algo_names);

func_nums_p2 = 11:20;
n_funcs      = numel(func_nums_p2);
func_labels_p2 = {
    'F11:Hybrid1 (N=3)',
    'F12:Hybrid2 (N=4)',
    'F13:Hybrid3 (N=5)',
    'F14:Hybrid4 (N=3)',
    'F15:Hybrid5 (N=4)',
    'F16:Hybrid6 (N=4)',
    'F17:Hybrid6 (N=5)',
    'F18:Hybrid7 (N=5)',
    'F19:Hybrid6 (N=5)',
    'F20:Hybrid6 (N=6)',
};
func_types_p2 = repmat({'Hybrid'},1,10);

dims=[10,30,50]; n_dims=3;

results_p2 = zeros(n_funcs,n_dims,n_algos,n_runs);
conv_p2    = zeros(n_funcs,n_dims,n_algos,T_max,n_runs);

t_start=tic;
fprintf('\n%s\n  CEC-2017 — Part 2: F11-F20 (Hybrid)\n',repmat('=',1,80));
fprintf('  Algorithms:%d | Dims:[10 30 50] | Runs:%d\n%s\n\n',...
        n_algos,n_runs,repmat('=',1,80));

for di=1:n_dims
    D=dims(di); lb=-100*ones(1,D); ub=100*ones(1,D);
    fprintf('%s\n  D=%d\n%s\n',repmat('-',1,80),D,repmat('-',1,80));
    for fi=1:n_funcs
        fnum=func_nums_p2(fi); f_star=fnum*100;
        fprintf('\n  [%d/%d] %s | D=%d\n',fi,n_funcs,func_labels_p2{fi},D);
        obj=@(x) cec17_func(x,fnum,D);
        for ai=1:n_algos
            bv=zeros(n_runs,1); cv=zeros(T_max,n_runs);
            for run=1:n_runs
                rng(run,'twister');
                [~,fv,c]=run_algo(algo_names{ai},obj,lb,ub,N,T_max,T_B,NY,flags);
                bv(run)=fv-f_star; cv(:,run)=c-f_star;
            end
            results_p2(fi,di,ai,:)=bv; conv_p2(fi,di,ai,:,:)=cv;
            fprintf('    %-8s | Mean=%10.4e | Std=%10.4e\n',...
                    algo_names{ai},mean(bv),std(bv));
        end
    end
end

elapsed=toc(t_start);
fprintf('\n  Part 2 done: %.1f min\n',elapsed/60);
save(fullfile(results_dir,'results_part2.mat'),...
     'results_p2','conv_p2','algo_names','func_nums_p2','func_labels_p2',...
     'func_types_p2','dims','n_runs','T_max','N','flags','elapsed');
fprintf('  Saved to: %s\n\n',results_dir);
diary off;

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
