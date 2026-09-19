%% run_09_ablation_study.m
%  GOA v3.2 ABLATION STUDY
%  --------------------------------------------------
%  Tests 8 GOA variants — each removes one key component —
%  to identify which mechanisms drive performance.
%
%  Variants:
%    V1: GOA_full      — complete v3.2 (baseline)
%    V2: GOA_noButta   — disable Butta ceremony (no renewal)
%    V3: GOA_noYuba    — disable Yuba advisory archive
%    V4: GOA_noDE      — G2-G3 use random step instead of DE/pbest/1
%    V5: GOA_noAPM     — fixed F=0.5, CR=0.5 (no SHADE memory)
%    V6: GOA_noEigen   — no eigenvector crossover
%    V7: GOA_noOBL     — no Opposition-Based Learning in G6
%    V8: GOA_noAlly    — disable inter-Gogessa ally sharing
%
%  Test functions (representative mix):
%    F1: Rastrigin D=30      (classical multimodal)
%    F2: Schwefel D=30       (deceptive multimodal)
%    F3: SR-Rastrigin D=30   (shifted+rotated)
%    F4: Welded Beam         (constrained engineering)
%    F5: CEC17-F17 D=30      (hybrid composition)
%
%  20 runs | T_max=500 | N=50
%  ~2-3 hours total
%  Output: C:\GOA PROJECT\Results\Ablation Study\
% -----------------------------------------------------------------------
clear; clc; close all;

code_dir    = 'C:\GOA PROJECT\GOA_MATLAB_Code';
results_dir = 'C:\GOA PROJECT\Results\Ablation Study';
addpath(code_dir); cd(code_dir); clear functions
if ~exist(results_dir,'dir'), mkdir(results_dir); end
diary(fullfile(results_dir,'Console_Output.txt')); diary on;

N=50; T_max=500; T_B=25; NY=10; n_runs=20;

% Full GOA v3.2 flags (baseline)
flags_full.use_butta=true; flags_full.use_yuba=true;
flags_full.use_topology=true; flags_full.use_siinqee=true;
flags_full.use_gumii=true; flags_full.use_incremental=true;
flags_full.use_g5shadow=true; flags_full.beta_val=0.1;
flags_full.problem_type='static';
flags_full.use_apm=true; flags_full.use_eigen=true;

%% ===== DEFINE 8 VARIANTS =====
variants(1).name='GOA_full';         variants(1).flags=flags_full;
variants(1).desc='Complete v3.2 (baseline)';

f2=flags_full; f2.use_butta=false;
variants(2).name='GOA_noButta';      variants(2).flags=f2;
variants(2).desc='No Butta ceremony (no population renewal)';

f3=flags_full; f3.use_yuba=false;
variants(3).name='GOA_noYuba';       variants(3).flags=f3;
variants(3).desc='No Yuba advisory archive';

% V4: No Gamme differential update — centroid step instead
f4=flags_full; f4.use_apm=false; f4.use_de_g23=false;
variants(4).name='GOA_noGammeDiff';  variants(4).flags=f4;
variants(4).desc='No Gamme differential update (centroid step)';

f5=flags_full; f5.use_apm=false;
variants(5).name='GOA_noAPM';        variants(5).flags=f5;
variants(5).desc='No Adaptive Parameter Memory (fixed F=0.5, CR=0.5)';

f6=flags_full; f6.use_eigen=false;
variants(6).name='GOA_noEigenspace'; variants(6).flags=f6;
variants(6).desc='No Gumii Eigenspace Exploration';

f7=flags_full; f7.use_obl=false;
variants(7).name='GOA_noGadaScout';  variants(7).flags=f7;
variants(7).desc='No Gada Opposition Scouting';

f8=flags_full; f8.use_topology=false;
variants(8).name='GOA_noAlly';       variants(8).flags=f8;
variants(8).desc='No Ally Sharing between Gogessas';

n_variants = numel(variants);

%% ===== DEFINE TEST FUNCTIONS =====
D=30; lam=1e8;
rng(42,'twister');
o_sch = -500*0.8+(2*500*0.8)*rand(1,D);
rng(7,'twister');
[R_mat,~]=qr(randn(D));
if det(R_mat)<0, R_mat(:,1)=-R_mat(:,1); end

tests(1).name='Rastrigin D=30';
tests(1).f=@(x) 10*D+sum(x.^2-10*cos(2*pi*x));
tests(1).lb=-5.12*ones(1,D); tests(1).ub=5.12*ones(1,D);

tests(2).name='Schwefel D=30';
tests(2).f=@(x) 418.9829*D-sum(x.*sin(sqrt(abs(x))));
tests(2).lb=-500*ones(1,D); tests(2).ub=500*ones(1,D);

tests(3).name='SR-Rastrigin D=30';
tests(3).f=@(x) 10*D+sum((R_mat*(x-o_sch)').^2-10*cos(2*pi*(R_mat*(x-o_sch)')));
tests(3).lb=-5.12*ones(1,D); tests(3).ub=5.12*ones(1,D);

tests(4).name='Welded Beam';
tests(4).f=@(x) welded_beam_obj(x,lam);
tests(4).lb=[0.1,0.1,0.1,0.1]; tests(4).ub=[2.0,10.0,10.0,2.0];

% CEC17-F17 D=30 (if cec17_func available)
try
    test_f17=@(x) cec17_func(x,17,30);
    test_f17(zeros(1,30));  % test call
    tests(5).name='CEC17-F17 D=30';
    tests(5).f=test_f17;
    tests(5).lb=-100*ones(1,30); tests(5).ub=100*ones(1,30);
catch
    % Fallback: Griewank D=30
    tests(5).name='Griewank D=30';
    tests(5).f=@(x) sum(x.^2)/4000-prod(cos(x./sqrt(1:D)))+1;
    tests(5).lb=-600*ones(1,D); tests(5).ub=600*ones(1,D);
end

n_tests = numel(tests);

%% ===== PRE-ALLOCATE =====
results = zeros(n_tests, n_variants, n_runs);  % raw fitness values
conv    = zeros(n_tests, n_variants, T_max);   % mean convergence

%% ===== MAIN LOOP =====
t_start=tic;
fprintf('\n%s\n  GOA ABLATION STUDY — v3.2\n',repmat('=',1,75));
fprintf('  Variants: %d | Functions: %d | Runs: %d | T_max: %d\n',...
        n_variants,n_tests,n_runs,T_max);
fprintf('%s\n\n',repmat('=',1,75));

for fi=1:n_tests
    fprintf('%s\n  TEST: %s\n%s\n',repmat('-',1,75),tests(fi).name,repmat('-',1,75));
    for vi=1:n_variants
        conv_acc=zeros(T_max,1);
        for run=1:n_runs
            rng(run,'twister');
            [~,fv,cv]=GOA_fixed(tests(fi).f, tests(fi).lb, tests(fi).ub,...
                                 N, T_max, T_B, NY, variants(vi).flags);
            results(fi,vi,run)=fv;
            conv_acc=conv_acc+cv;
        end
        conv(fi,vi,:)=conv_acc/n_runs;
        fprintf('  %-18s | Mean=%11.4e | Std=%10.4e\n',...
                variants(vi).name,...
                mean(squeeze(results(fi,vi,:))),...
                std(squeeze(results(fi,vi,:))));
    end
    fprintf('\n');
end

elapsed=toc(t_start);
fprintf('\n%s\n  Total: %.1f min\n%s\n',repmat('=',1,75),elapsed/60,repmat('=',1,75));

%% ===== DEGRADATION ANALYSIS =====
fprintf('\n%s\n  DEGRADATION TABLE (vs GOA_full baseline)\n%s\n',...
        repmat('=',1,75),repmat('=',1,75));

means=zeros(n_tests,n_variants);
for fi=1:n_tests
    for vi=1:n_variants
        means(fi,vi)=mean(squeeze(results(fi,vi,:)));
    end
end

% Normalized degradation: (variant - full) / full * 100%
baseline=means(:,1);
fprintf('\n  %-18s','Function');
for vi=2:n_variants
    fprintf('  %-14s',variants(vi).name);
end
fprintf('\n  %s\n',repmat('-',1,18+(n_variants-1)*16));

for fi=1:n_tests
    fprintf('  %-18s',tests(fi).name);
    for vi=2:n_variants
        deg=(means(fi,vi)-means(fi,1))/(abs(means(fi,1))+eps)*100;
        if deg>0, marker='+'; else, marker=' '; end
        fprintf('  %s%11.1f%%  ',marker,deg);
    end
    fprintf('\n');
end

% Mean degradation per variant (higher = component matters more)
fprintf('\n  %-18s','MEAN DEGRADATION');
for vi=2:n_variants
    avg_deg=mean((means(:,vi)-means(:,1))./(abs(means(:,1))+eps)*100);
    fprintf('  %+11.1f%%  ',avg_deg);
end
fprintf('\n\n');

fprintf('  Interpretation: Higher degradation = component is MORE important\n');
fprintf('  Negative degradation = removing that component accidentally HELPED\n\n');

%% ===== RANK TABLE =====
fprintf('%s\n  VARIANT RANKING (lower rank = better)\n%s\n',...
        repmat('=',1,75),repmat('=',1,75));

mean_ranks=zeros(n_variants,1);
for fi=1:n_tests
    [~,ord]=sort(means(fi,:));
    for vi=1:n_variants
        mean_ranks(vi)=mean_ranks(vi)+find(ord==vi);
    end
end
mean_ranks=mean_ranks/n_tests;
[~,sr]=sort(mean_ranks);
for vi=1:n_variants
    fprintf('  %d. %-18s Rank=%.2f | Desc: %s\n',...
            vi,variants(sr(vi)).name,mean_ranks(sr(vi)),variants(sr(vi)).desc);
end

%% ===== SAVE =====
save(fullfile(results_dir,'ablation_results.mat'),...
     'results','conv','means','mean_ranks',...
     'variants','tests','n_runs','T_max','N','elapsed');

fprintf('\n  Saved to: %s\n',results_dir);
diary off;

%% ===== LOCAL FUNCTION =====
function f=welded_beam_obj(x,lam)
    h=x(1);l=x(2);t=x(3);b=x(4);
    P=6000;L=14;E=30e6;G=12e6;
    tau_max=13600;sigma_max=30000;delta_max=0.25;
    M=P*(L+l/2); R=sqrt(l^2/4+((h+t)/2)^2);
    J=2*(sqrt(2)*h*l*(l^2/12+((h+t)/2)^2));
    tau1=P/(sqrt(2)*h*l); tau2=M*R/J;
    tau=sqrt(tau1^2+2*tau1*tau2*l/(2*R)+tau2^2);
    sigma=6*P*L/(b*t^2); delta=4*P*L^3/(E*b*t^3);
    Pc=(4.013*E*sqrt(t^2*b^6/36)/L^2)*(1-t/(2*L)*sqrt(E/(4*G)));
    f0=1.10471*h^2*l+0.04811*t*b*(14+l);
    g(1)=tau-tau_max;g(2)=sigma-sigma_max;g(3)=h-b;
    g(4)=delta-delta_max;g(5)=P-Pc;
    f=f0+lam*sum(max(0,g).^2);
end
