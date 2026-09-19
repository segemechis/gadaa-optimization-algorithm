%% run_08a_CEC2017_part1.m
%  CEC-2017 Benchmark — Part 1 of 3
%  F1, F3-F10 (9 functions: Unimodal + Basic Multimodal)
%  F2 excluded (ill-conditioned in CEC-2017 standard)
%  D=10,30,50 | 13 algorithms | N=50 | T_max=500 | 30 runs
%  Output: C:\GOA PROJECT\Results\CEC2017 Benchmark\part1\
% -----------------------------------------------------------------------
clear; clc; close all;

code_dir    = 'C:\GOA PROJECT\GOA_MATLAB_Code';
results_dir = 'C:\GOA PROJECT\Results\CEC2017 Benchmark\part1';
addpath(code_dir); cd(code_dir); clear functions
if ~exist(results_dir,'dir'), mkdir(results_dir); end
diary(fullfile(results_dir,'Console_Output_Part1.txt')); diary on;

N=50; T_max=500; T_B=25; NY=10; n_runs=30;

flags.use_butta=true; flags.use_yuba=true; flags.use_topology=true;
flags.use_siinqee=true; flags.use_gumii=true; flags.use_incremental=true;
flags.use_g5shadow=true; flags.beta_val=0.1;
flags.problem_type='static';

algo_names={'GOA','PSO','GWO','WOA','DE','RIME','MVO','TLBO',...
            'SO','SHO','SCSO','SLO','CCO'};
n_algos=numel(algo_names);

% F1, F3-F10 (skip F2)
func_nums_p1 = [1, 3, 4, 5, 6, 7, 8, 9, 10];
n_funcs      = numel(func_nums_p1);
func_labels_p1 = {
    'F1:BentCigar (Unimodal)',
    'F3:Zakharov (Unimodal)',
    'F4:Rosenbrock (Multimodal)',
    'F5:Rastrigin (Multimodal)',
    'F6:ScafferF6 (Multimodal)',
    'F7:LunacekBiRast (Multimodal)',
    'F8:NonContRast (Multimodal)',
    'F9:Levy (Multimodal)',
    'F10:ModSchwefel (Multimodal)',
};
func_types_p1 = {'Unimodal','Unimodal','Multimodal','Multimodal','Multimodal',...
                 'Multimodal','Multimodal','Multimodal','Multimodal'};

dims=[10,30,50]; n_dims=3;

results_p1 = zeros(n_funcs,n_dims,n_algos,n_runs);
conv_p1    = zeros(n_funcs,n_dims,n_algos,T_max,n_runs);

t_start=tic;
fprintf('\n%s\n  CEC-2017 — Part 1: F1,F3-F10 (Unimodal + Multimodal)\n',...
        repmat('=',1,80));
fprintf('  Algorithms:%d | Dims:[10 30 50] | Runs:%d\n%s\n\n',...
        n_algos,n_runs,repmat('=',1,80));

for di=1:n_dims
    D=dims(di); lb=-100*ones(1,D); ub=100*ones(1,D);
    fprintf('%s\n  D=%d\n%s\n',repmat('-',1,80),D,repmat('-',1,80));
    for fi=1:n_funcs
        fnum=func_nums_p1(fi); f_star=fnum*100;
        fprintf('\n  [%d/%d] %s | D=%d | f*=%d\n',fi,n_funcs,func_labels_p1{fi},D,f_star);
        obj=@(x) cec17_func(x,fnum,D);
        for ai=1:n_algos
            bv=zeros(n_runs,1); cv=zeros(T_max,n_runs);
            for run=1:n_runs
                rng(run,'twister');
                [~,fv,c]=run_algo(algo_names{ai},obj,lb,ub,N,T_max,T_B,NY,flags);
                bv(run)=fv-f_star; cv(:,run)=c-f_star;
            end
            results_p1(fi,di,ai,:)=bv; conv_p1(fi,di,ai,:,:)=cv;
            fprintf('    %-8s | Mean=%10.4e | Std=%10.4e | Best=%10.4e\n',...
                    algo_names{ai},mean(bv),std(bv),min(bv));
        end
    end
end

elapsed=toc(t_start);
fprintf('\n  Part 1 done: %.1f min\n',elapsed/60);
save(fullfile(results_dir,'results_part1.mat'),...
     'results_p1','conv_p1','algo_names','func_nums_p1','func_labels_p1',...
     'func_types_p1','dims','n_runs','T_max','N','flags','elapsed');
fprintf('  Saved to: %s\n\n',results_dir);
diary off;

%% LOCAL FUNCTIONS
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
