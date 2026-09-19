%% run_08c_CEC2017_part3.m
%  CEC-2017 Benchmark — Part 3 of 3
%  F21-F30 (10 Composition Functions)
%  D=10,30,50 | 13 algorithms | N=50 | T_max=500 | 30 runs
%  Output: C:\GOA PROJECT\Results\CEC2017 Benchmark\part3\
% -----------------------------------------------------------------------
clear; clc; close all;

code_dir    = 'C:\GOA PROJECT\GOA_MATLAB_Code';
results_dir = 'C:\GOA PROJECT\Results\CEC2017 Benchmark\part3';
addpath(code_dir); cd(code_dir); clear functions
if ~exist(results_dir,'dir'), mkdir(results_dir); end
diary(fullfile(results_dir,'Console_Output_Part3.txt')); diary on;

N=50; T_max=500; T_B=25; NY=10; n_runs=30;

flags.use_butta=true; flags.use_yuba=true; flags.use_topology=true;
flags.use_siinqee=true; flags.use_gumii=true; flags.use_incremental=true;
flags.use_g5shadow=true; flags.beta_val=0.1;
flags.problem_type='static';

algo_names={'GOA','PSO','GWO','WOA','DE','RIME','MVO','TLBO',...
            'SO','SHO','SCSO','SLO','CCO'};
n_algos=numel(algo_names);

func_nums_p3 = 21:30;
n_funcs      = numel(func_nums_p3);
func_labels_p3 = {
    'F21:Composition1 (N=3)',
    'F22:Composition2 (N=3)',
    'F23:Composition3 (N=4)',
    'F24:Composition4 (N=4)',
    'F25:Composition5 (N=5)',
    'F26:Composition6 (N=5)',
    'F27:Composition7 (N=6)',
    'F28:Composition8 (N=3)',
    'F29:Composition9 (N=3)',
    'F30:Composition10 (N=3)',
};
func_types_p3 = repmat({'Composition'},1,10);

dims=[10,30,50]; n_dims=3;

results_p3 = zeros(n_funcs,n_dims,n_algos,n_runs);
conv_p3    = zeros(n_funcs,n_dims,n_algos,T_max,n_runs);

t_start=tic;
fprintf('\n%s\n  CEC-2017 — Part 3: F21-F30 (Composition)\n',repmat('=',1,80));
fprintf('  Algorithms:%d | Dims:[10 30 50] | Runs:%d\n%s\n\n',...
        n_algos,n_runs,repmat('=',1,80));

for di=1:n_dims
    D=dims(di); lb=-100*ones(1,D); ub=100*ones(1,D);
    fprintf('%s\n  D=%d\n%s\n',repmat('-',1,80),D,repmat('-',1,80));
    for fi=1:n_funcs
        fnum=func_nums_p3(fi); f_star=fnum*100;
        fprintf('\n  [%d/%d] %s | D=%d\n',fi,n_funcs,func_labels_p3{fi},D);
        obj=@(x) cec17_func(x,fnum,D);
        for ai=1:n_algos
            bv=zeros(n_runs,1); cv=zeros(T_max,n_runs);
            for run=1:n_runs
                rng(run,'twister');
                [~,fv,c]=run_algo(algo_names{ai},obj,lb,ub,N,T_max,T_B,NY,flags);
                bv(run)=fv-f_star; cv(:,run)=c-f_star;
            end
            results_p3(fi,di,ai,:)=bv; conv_p3(fi,di,ai,:,:)=cv;
            fprintf('    %-8s | Mean=%10.4e | Std=%10.4e\n',...
                    algo_names{ai},mean(bv),std(bv));
        end
    end
end

elapsed=toc(t_start);
fprintf('\n  Part 3 done: %.1f min\n',elapsed/60);
save(fullfile(results_dir,'results_part3.mat'),...
     'results_p3','conv_p3','algo_names','func_nums_p3','func_labels_p3',...
     'func_types_p3','dims','n_runs','T_max','N','flags','elapsed');
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
