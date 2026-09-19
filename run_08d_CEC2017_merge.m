%% run_08d_CEC2017_merge.m
%  Merge CEC-2017 Parts 1+2+3 → full statistics + figures
%  Run AFTER all three parts complete.
%  Output: C:\GOA PROJECT\Results\CEC2017 Benchmark\
% -----------------------------------------------------------------------
clear; clc; close all;

base_dir  = 'C:\GOA PROJECT\Results\CEC2017 Benchmark';
part1_dir = fullfile(base_dir,'part1');
part2_dir = fullfile(base_dir,'part2');
part3_dir = fullfile(base_dir,'part3');
fig_dir   = fullfile(base_dir,'Figures');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end

diary(fullfile(base_dir,'Console_Output_Full.txt')); diary on;

fprintf('Loading parts...\n');
p1=load(fullfile(part1_dir,'results_part1.mat'));
p2=load(fullfile(part2_dir,'results_part2.mat'));
p3=load(fullfile(part3_dir,'results_part3.mat'));

algo_names = p1.algo_names;
n_algos    = numel(algo_names);
dims       = p1.dims;
n_dims     = numel(dims);
n_runs     = p1.n_runs;
T_max      = p1.T_max;

% Merge all 29 functions — use cell array concatenation for string arrays
func_nums   = [p1.func_nums_p1,  p2.func_nums_p2,  p3.func_nums_p3];
func_labels = [p1.func_labels_p1(:)', p2.func_labels_p2(:)', p3.func_labels_p3(:)'];
func_types  = [p1.func_types_p1(:)',  p2.func_types_p2(:)',  p3.func_types_p3(:)'];
n_funcs     = numel(func_nums);

results = cat(1, p1.results_p1, p2.results_p2, p3.results_p3);
conv    = cat(1, p1.conv_p1,    p2.conv_p2,    p3.conv_p3);

colors_rgb=[0.00 0.40 0.80;0.80 0.20 0.20;0.10 0.60 0.10;
            0.60 0.10 0.60;0.90 0.50 0.00;0.10 0.70 0.70;
            0.40 0.40 0.40;0.85 0.33 0.10;0.50 0.00 0.50;
            0.00 0.60 0.50;0.70 0.70 0.00;0.80 0.00 0.00;0.00 0.00 0.80];

%% ===== FRIEDMAN RANK =====
fprintf('\n%s\n  FRIEDMAN RANK — ALL 29 FUNCTIONS\n%s\n',...
        repmat('=',1,80),repmat('=',1,80));

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
fprintf('\n  COMBINED (29 functions, 3 dimensions):\n');
[~,sr]=sort(rank_combined);
for ai=1:n_algos
    fprintf('    %2d. %-8s  Rank=%.2f\n',ai,algo_names{sr(ai)},rank_combined(sr(ai)));
end

%% ===== TYPE ANALYSIS (D=30) =====
fprintf('\n%s\n  TYPE ANALYSIS — D=30\n%s\n',repmat('=',1,80),repmat('=',1,80));
di30=find(dims==30);
type_names={'Unimodal','Multimodal','Hybrid','Composition'};
type_masks={strcmp(func_types,'Unimodal'),...
            strcmp(func_types,'Multimodal'),...
            strcmp(func_types,'Hybrid'),...
            strcmp(func_types,'Composition')};

for ti=1:4
    mask=type_masks{ti};
    if ~any(mask), continue; end
    M_t=zeros(sum(mask),n_algos);
    r=0;
    for fi=1:n_funcs
        if mask(fi)
            r=r+1;
            for ai=1:n_algos
                M_t(r,ai)=mean(squeeze(results(fi,di30,ai,:)));
            end
        end
    end
    R_t=zeros(size(M_t,1),n_algos);
    for fi=1:size(M_t,1)
        [~,ord]=sort(M_t(fi,:));
        r_tmp=zeros(1,n_algos);
        for ri=1:n_algos, r_tmp(ord(ri))=ri; end
        R_t(fi,:)=r_tmp;
    end
    rank_t=mean(R_t,1);
    fprintf('\n  %s (%d functions):\n',type_names{ti},sum(mask));
    [~,sr]=sort(rank_t);
    for ai=1:min(5,n_algos)
        fprintf('    %d. %-8s Rank=%.2f\n',ai,algo_names{sr(ai)},rank_t(sr(ai)));
    end
end

%% ===== FRIEDMAN TEST (D=30) =====
di30=find(dims==30);
M30=zeros(n_funcs,n_algos);
for fi=1:n_funcs
    for ai=1:n_algos
        M30(fi,ai)=mean(squeeze(results(fi,di30,ai,:)));
    end
end
R30=zeros(n_funcs,n_algos);
for fi=1:n_funcs
    [~,ord]=sort(M30(fi,:));
    r_tmp=zeros(1,n_algos);
    for ri=1:n_algos, r_tmp(ord(ri))=ri; end
    R30(fi,:)=r_tmp;
end
R30_mean=mean(R30,1);
k=n_algos; n_obs=n_funcs;
chi2=12*n_obs/(k*(k+1))*(sum(R30_mean.^2)-k*(k+1)^2/4);
p_frd=1-chi2cdf(chi2,k-1);
fprintf('\n  Friedman test D=30: chi2=%.2f, p=%.6f\n',chi2,p_frd);

% Holm post-hoc
goa_r=R30_mean(1);
z_vals=zeros(n_algos-1,1); p_vals=zeros(n_algos-1,1);
for ci=1:n_algos-1
    z_vals(ci)=(R30_mean(ci+1)-goa_r)/sqrt(k*(k+1)/(6*n_obs));
    p_vals(ci)=2*(1-normcdf(abs(z_vals(ci))));
end
fprintf('\n  Holm post-hoc (GOA vs others, D=30):\n');
[~,po]=sort(p_vals);
rej=false(n_algos-1,1);
for ci=1:numel(p_vals)
    if p_vals(po(ci))<0.05/(numel(p_vals)-ci+1), rej(ci)=true; else, break; end
end
for ci=1:n_algos-1
    tag='n.s.'; if rej(ci), tag='*'; end
    fprintf('    GOA vs %-8s | z=%+.3f | p=%.4f %s\n',...
            algo_names{ci+1},z_vals(ci),p_vals(ci),tag);
end

%% ===== CSV =====
fprintf('\n  Saving CSV...\n');
for di=1:n_dims
    fid=fopen(fullfile(base_dir,sprintf('Table_CEC2017_D%d.csv',dims(di))),'w');
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
    fclose(fid);
end

%% ===== FIGURES =====
fprintf('  Generating figures...\n');

% Combined rank bar
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
title('CEC-2017 Combined Friedman Rank (29 functions)','FontSize',13);
ylim([0 n_algos+1]); grid on; box on; hold off;
saveas(fig,fullfile(fig_dir,'Fig_Friedman_Combined.png')); close(fig);

% Scalability
fig=figure('Visible','off','Position',[100 100 800 450]);
hold on;
for ai=1:n_algos
    plot(dims,rank_per_dim(ai,:),'-o','Color',colors_rgb(ai,:),'LineWidth',1.5);
end
xlabel('Dimension D','FontSize',12); ylabel('Mean Rank','FontSize',12);
title('CEC-2017 Rank vs Dimension','FontSize',13);
legend(algo_names,'Location','eastoutside','FontSize',8);
set(gca,'XTick',dims); ylim([0 n_algos+1]); grid on; box on; hold off;
saveas(fig,fullfile(fig_dir,'Fig_Scalability.png')); close(fig);

% Convergence curves D=30 (selected functions)
key_funcs = [1,4,9,11,15,21,25,28]; % representative from each type
for ki=1:numel(key_funcs)
    fi=key_funcs(ki);
    if fi>n_funcs, continue; end
    fig=figure('Visible','off','Position',[100 100 800 450]);
    hold on;
    for ai=1:n_algos
        mc=mean(squeeze(conv(fi,di30,ai,:,:)),2);
        semilogy(1:T_max,max(mc,1e-20),'-','Color',colors_rgb(ai,:),'LineWidth',1.2);
    end
    xlabel('Iteration'); ylabel('Error (f-f*)');
    title(sprintf('%s D=30',func_labels{fi}));
    legend(algo_names,'Location','northeast','FontSize',7);
    set(gca,'YScale','log'); grid on; box on; hold off;
    saveas(fig,fullfile(fig_dir,sprintf('Fig_Conv_F%d_D30.png',func_nums(fi)))); close(fig);
end

%% ===== SAVE =====
save(fullfile(base_dir,'full_CEC2017_results.mat'),...
     'results','conv','algo_names','func_nums','func_labels','func_types',...
     'dims','n_runs','T_max','rank_per_dim','rank_combined','chi2','p_frd');

fprintf('\n  Merge complete. All saved to: %s\n\n',base_dir);
diary off;
