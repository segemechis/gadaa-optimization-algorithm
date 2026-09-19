function [pop, Y_archive, Y_fitness, Y_cv, x_best, f_best, cv_best] = ...
    GOA_butta_deb(pop, Y_archive, Y_fitness, Y_cv, x_best, f_best, cv_best, ...
                  lb, ub, K, NY, safe_eval)
% GOA_BUTTA_DEB  Butta transition with Deb feasibility-guided archive and election.

    d    = numel(lb);
    n_G1 = 2;

    % 1. Archive current Abba Gada
    [Y_archive, Y_fitness, Y_cv] = yuba_insert_deb(...
        Y_archive, Y_fitness, Y_cv, x_best, f_best, cv_best, NY);

    % 2. Grade promotion
    for k = 1:K
        for g = 6:-1:2
            pop(k).grade{g}    = pop(k).grade{g-1};
            pop(k).grade_f{g}  = pop(k).grade_f{g-1};
            pop(k).grade_cv{g} = pop(k).grade_cv{g-1};
        end
        % New Dabale
        for j = 1:n_G1
            x_new = lb + rand(1,d).*(ub-lb);
            [f_new,cv_new] = safe_eval(x_new);
            pop(k).grade{1}(j,:)   = x_new;
            pop(k).grade_f{1}(j)   = f_new;
            pop(k).grade_cv{1}(j)  = cv_new;
        end
    end

    % 3. New Abba Gada election using Deb
    f_best  = inf;  cv_best = inf;
    x_best  = pop(1).grade{6}(1,:);
    for k = 1:K
        for j = 1:size(pop(k).grade{6},1)
            fj  = pop(k).grade_f{6}(j);
            cvj = pop(k).grade_cv{6}(j);
            if deb_compare(fj,cvj,f_best,cv_best)
                f_best  = fj;  cv_best = cvj;
                x_best  = pop(k).grade{6}(j,:);
            end
        end
    end
end

function [Y_arc, Y_fit, Y_cv] = yuba_insert_deb(Y_arc, Y_fit, Y_cv, x, f, cv, NY)
    if isempty(Y_arc)
        Y_arc = x;  Y_fit = f;  Y_cv = cv;
        return;
    end
    n = size(Y_arc,1);
    if n < NY
        Y_arc = [Y_arc; x];
        Y_fit = [Y_fit; f];
        Y_cv  = [Y_cv;  cv];
    else
        % Find worst entry using Deb (replace if x is better)
        worst_idx = 1;
        for i = 2:n
            if deb_compare(Y_fit(i),Y_cv(i),Y_fit(worst_idx),Y_cv(worst_idx))
                % i is better than worst_idx, so worst_idx stays as worst
            else
                worst_idx = i;
            end
        end
        if deb_compare(f,cv,Y_fit(worst_idx),Y_cv(worst_idx))
            Y_arc(worst_idx,:) = x;
            Y_fit(worst_idx)   = f;
            Y_cv(worst_idx)    = cv;
        end
    end
    % Sort by Deb order (feasible first, then by objective)
    feasible = Y_cv <= 1e-10;
    f_sort   = Y_fit;
    f_sort(~feasible) = f_sort(~feasible) + 1e12;
    [~,idx]  = sort(f_sort);
    Y_arc    = Y_arc(idx,:);
    Y_fit    = Y_fit(idx);
    Y_cv     = Y_cv(idx);
end
