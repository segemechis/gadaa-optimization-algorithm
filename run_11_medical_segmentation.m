%% run_11_medical_segmentation.m
%  Multi-Level Medical Image Thresholding using Kapur's Entropy
%  --------------------------------------------------
%  5 biomedical images (built-in MATLAB Image Processing Toolbox):
%    I1: mri.tif        — Brain MRI
%    I2: spine.tif      — Spinal X-ray
%    I3: hestain.png    — Histology H&E stain
%    I4: AT3_1m4_01.tif — Fluorescence microscopy
%    I5: tissue.png     — Tissue microscopy
%
%  Threshold levels: K = 3, 5, 7, 10
%  Metrics: Kapur entropy, PSNR, SSIM, FSIM (approx), runtime
%  13 algorithms | N=50 | T_max=200 | 30 runs
%  Output: C:\GOA PROJECT\Results\Medical Segmentation\
% -----------------------------------------------------------------------
clear; clc; close all;

code_dir    = 'C:\GOA PROJECT\GOA_MATLAB_Code';
results_dir = 'C:\GOA PROJECT\Results\Medical Segmentation';
img_dir     = fullfile(matlabroot,'toolbox','images','imdata');

addpath(code_dir); cd(code_dir); clear functions
if ~exist(results_dir,'dir'), mkdir(results_dir); end
diary(fullfile(results_dir,'Console_Output.txt')); diary on;

N=50; T_max=200; T_B=25; NY=10; n_runs=30;

flags.use_butta=true; flags.use_yuba=true; flags.use_topology=true;
flags.use_siinqee=true; flags.use_gumii=true; flags.use_incremental=true;
flags.use_g5shadow=true; flags.beta_val=0.1;
flags.problem_type='static'; flags.use_apm=true; flags.use_eigen=true;

algo_names={'GOA','PSO','GWO','WOA','DE','RIME','MVO','TLBO',...
            'SO','SHO','SCSO','SLO','CCO'};
n_algos=numel(algo_names);
colors_rgb=[0.00 0.40 0.80;0.80 0.20 0.20;0.10 0.60 0.10;
            0.60 0.10 0.60;0.90 0.50 0.00;0.10 0.70 0.70;
            0.40 0.40 0.40;0.85 0.33 0.10;0.50 0.00 0.50;
            0.00 0.60 0.50;0.70 0.70 0.00;0.80 0.00 0.00;0.00 0.00 0.80];

%% ===== IMAGES =====
img_files = {'mri.tif','spine.tif','hestain.png','AT3_1m4_01.tif','tissue.png'};
img_names = {'Brain MRI','Spinal X-ray','Histology H&E','Fluorescence Microscopy','Tissue Microscopy'};
n_imgs    = numel(img_files);

% Pre-load and convert all images to grayscale uint8
fprintf('Loading images...\n');
imgs = cell(n_imgs,1);
for ii=1:n_imgs
    raw = imread(fullfile(img_dir, img_files{ii}));
    if size(raw,3)==3
        raw = rgb2gray(raw);
    end
    % Take first slice if 3D (mri.tif has multiple slices)
    if ndims(raw)==3
        raw = raw(:,:,1);
    end
    imgs{ii} = raw;
    fprintf('  %s: %dx%d\n', img_names{ii}, size(raw,1), size(raw,2));
end

%% ===== THRESHOLD LEVELS =====
K_levels = [3, 5, 7, 10];
n_K      = numel(K_levels);

%% ===== PRE-ALLOCATE =====
% Kapur entropy achieved (higher = better segmentation quality)
entropy_result = zeros(n_imgs, n_K, n_algos, n_runs);
% PSNR: Peak Signal-to-Noise Ratio (higher = better)
psnr_result    = zeros(n_imgs, n_K, n_algos, n_runs);
% SSIM: Structural Similarity Index (higher = better, max=1)
ssim_result    = zeros(n_imgs, n_K, n_algos, n_runs);
% Best thresholds found
thresh_result  = cell(n_imgs, n_K, n_algos);

%% ========================================================================
t_start=tic;
fprintf('\n%s\n  MEDICAL IMAGE THRESHOLDING — Kapur Entropy\n',repmat('=',1,80));
fprintf('  Images: %d | Threshold levels: %d | Algorithms: %d | Runs: %d\n',...
        n_imgs,n_K,n_algos,n_runs);
fprintf('%s\n\n',repmat('=',1,80));

for ii=1:n_imgs
    img = imgs{ii};
    [h_img, w_img] = size(img);
    hist_img = imhist(img) / numel(img);  % normalized histogram

    fprintf('%s\n  IMAGE: %s\n%s\n',repmat('-',1,80),img_names{ii},repmat('-',1,80));

    for ki=1:n_K
        K = K_levels(ki);
        D = K - 1;  % number of threshold variables
        fprintf('\n  K=%d thresholds (D=%d):\n', K, D);

        % Bounds: thresholds in [1, 254], must be ordered
        lb_t = ones(1,D);
        ub_t = 254*ones(1,D);

        % Objective: maximize Kapur entropy (minimize negative)
        obj = @(x) -kapur_entropy(sort(round(x)), hist_img);

        for ai=1:n_algos
            t_algo=tic;
            best_thresh_runs = zeros(n_runs, D);

            for run=1:n_runs
                rng(run,'twister');
                [xb, fv, ~] = run_algo(algo_names{ai}, obj, lb_t, ub_t,...
                                       N, T_max, T_B, NY, flags);
                thresholds = sort(round(xb));
                thresholds = max(1, min(254, thresholds));
                % Ensure strictly increasing
                for di=2:D
                    if thresholds(di) <= thresholds(di-1)
                        thresholds(di) = thresholds(di-1)+1;
                    end
                end
                thresholds = min(thresholds, 254);

                entropy_result(ii,ki,ai,run) = -fv;
                best_thresh_runs(run,:) = thresholds;

                % Compute segmented image metrics
                seg_img = apply_thresholds(img, thresholds, K);
                psnr_result(ii,ki,ai,run)   = compute_psnr(img, seg_img);
                ssim_result(ii,ki,ai,run)   = ssim(seg_img, img);
            end

            % Store mean best thresholds
            thresh_result{ii,ki,ai} = mean(best_thresh_runs,1);

            fprintf('    %-8s | Entropy=%6.4f | PSNR=%6.2f | SSIM=%.4f | %.1fs\n',...
                    algo_names{ai},...
                    mean(squeeze(entropy_result(ii,ki,ai,:))),...
                    mean(squeeze(psnr_result(ii,ki,ai,:))),...
                    mean(squeeze(ssim_result(ii,ki,ai,:))),...
                    toc(t_algo));
        end
    end
    fprintf('\n');
end

elapsed = toc(t_start);
fprintf('\n%s\n  Total: %.1f min\n%s\n',repmat('=',1,80),elapsed/60,repmat('=',1,80));

%% ===== RANKING =====
fprintf('\n%s\n  FRIEDMAN RANK — KAPUR ENTROPY (averaged across images and K levels)\n',...
        repmat('=',1,80));

% Rank on entropy (higher is better → flip sign for ranking)
n_cases = n_imgs * n_K;
R_all   = zeros(n_cases, n_algos);
row = 0;
for ii=1:n_imgs
    for ki=1:n_K
        row = row+1;
        means = zeros(n_algos,1);
        for ai=1:n_algos
            means(ai) = mean(squeeze(entropy_result(ii,ki,ai,:)));
        end
        [~,ord] = sort(means,'descend');  % higher entropy = better
        r_tmp = zeros(1,n_algos);
        for ri=1:n_algos, r_tmp(ord(ri))=ri; end
        R_all(row,:) = r_tmp;
    end
end
rank_seg = mean(R_all,1)';
[~,sr] = sort(rank_seg);
fprintf('%s\n',repmat('=',1,80));
for ai=1:n_algos
    fprintf('  %2d. %-8s  Rank=%.2f  MeanEntropy=%.4f  MeanPSNR=%.2f  MeanSSIM=%.4f\n',...
            ai, algo_names{sr(ai)}, rank_seg(sr(ai)),...
            mean(mean(mean(entropy_result(:,:,sr(ai),:),4),2),1),...
            mean(mean(mean(psnr_result(:,:,sr(ai),:),4),2),1),...
            mean(mean(mean(ssim_result(:,:,sr(ai),:),4),2),1));
end

%% ===== K-LEVEL SCALABILITY =====
fprintf('\n%s\n  SCALABILITY WITH K (GOA vs top 3 competitors)\n%s\n',...
        repmat('=',1,80),repmat('=',1,80));
[~,top4] = sort(rank_seg);
top4 = top4(1:4);
fprintf('\n  Algorithm    ');
for ki=1:n_K, fprintf('  K=%-2d(ent)',K_levels(ki)); end
fprintf('\n  %s\n',repmat('-',1,14+n_K*12));
for t=1:4
    ai=top4(t);
    fprintf('  %-12s',algo_names{ai});
    for ki=1:n_K
        fprintf('  %9.4f',mean(mean(squeeze(entropy_result(:,ki,ai,:)),2)));
    end
    fprintf('\n');
end

%% ===== CSV TABLE =====
fprintf('\n  Saving CSV...\n');
fid=fopen(fullfile(results_dir,'Table_Segmentation_Full.csv'),'w');
fprintf(fid,'Image,K,Metric');
for ai=1:n_algos, fprintf(fid,',%s',algo_names{ai}); end
fprintf(fid,'\n');
for ii=1:n_imgs
    for ki=1:n_K
        % Entropy
        fprintf(fid,'%s,K=%d,Entropy(mean)',img_names{ii},K_levels(ki));
        for ai=1:n_algos
            fprintf(fid,',%.4f',mean(squeeze(entropy_result(ii,ki,ai,:))));
        end
        fprintf(fid,'\n,,Entropy(std)');
        for ai=1:n_algos
            fprintf(fid,',%.4f',std(squeeze(entropy_result(ii,ki,ai,:))));
        end
        fprintf(fid,'\n,,PSNR(mean)');
        for ai=1:n_algos
            fprintf(fid,',%.4f',mean(squeeze(psnr_result(ii,ki,ai,:))));
        end
        fprintf(fid,'\n,,SSIM(mean)');
        for ai=1:n_algos
            fprintf(fid,',%.4f',mean(squeeze(ssim_result(ii,ki,ai,:))));
        end
        fprintf(fid,'\n');
    end
end
fprintf(fid,'\nFriedman Rank,,');
for ai=1:n_algos, fprintf(fid,',%.2f',rank_seg(ai)); end
fprintf(fid,'\n');
fclose(fid);

%% ===== SAVE =====
save(fullfile(results_dir,'segmentation_results.mat'),...
     'entropy_result','psnr_result','ssim_result','thresh_result',...
     'rank_seg','algo_names','img_names','img_files','K_levels',...
     'n_runs','T_max','N','elapsed','imgs');

fprintf('\n  Saved to: %s\n  Medical Segmentation complete.\n\n',results_dir);
diary off;

%% ========================================================================
%                     OBJECTIVE & HELPER FUNCTIONS
%% ========================================================================

function H = kapur_entropy(thresholds, hist_p)
%KAPUR_ENTROPY  Kapur's entropy for multilevel thresholding.
%  thresholds: sorted integer threshold vector (K-1 values)
%  hist_p: normalized histogram of image (256 values)
    thresholds = round(thresholds);
    thresholds = max(1, min(254, thresholds));
    levels = [0; thresholds(:); 255];
    n_classes = numel(levels)-1;
    H = 0;
    for c=1:n_classes
        lo = levels(c)+1;
        hi = levels(c+1)+1;
        if lo > hi || hi > 256, continue; end
        p_c = sum(hist_p(lo:hi));
        if p_c < 1e-10, continue; end
        p_norm = hist_p(lo:hi) / p_c;
        p_norm = p_norm(p_norm > 0);
        H = H - sum(p_norm .* log(p_norm));
    end
end


function seg = apply_thresholds(img, thresholds, K)
%APPLY_THRESHOLDS  Segment image using threshold values.
    thresholds = sort(round(thresholds));
    thresholds = max(1, min(254, thresholds));
    seg = zeros(size(img), 'uint8');
    levels = [0; thresholds(:); 255];
    step = 255 / (K-1+eps);
    for c=1:K
        lo = levels(c);
        hi = levels(c+1);
        val = uint8(round((c-1)*step));
        mask = img >= lo & img <= hi;
        seg(mask) = val;
    end
end


function p = compute_psnr(orig, seg)
%COMPUTE_PSNR  PSNR between original and segmented image.
    orig = double(orig); seg = double(seg);
    mse = mean((orig(:)-seg(:)).^2);
    if mse < 1e-10, p = 100; return; end
    p = 10*log10(255^2/mse);
end


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
