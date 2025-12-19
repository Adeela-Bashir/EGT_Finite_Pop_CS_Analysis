clear; clc; close all;

%% ============================================================
%  SWITCH: subsidy scenario
%  ------------------------------------------------------------
%  subsidy_on = false  -> baseline: zealots pay same CH as H
%  subsidy_on = true   -> subsidised zealots: zealots' cost CH
%                         is effectively covered by government
% =============================================================
subsidy_on = true;   % <--- TOGGLE HERE (true / false)

%% -----------------------
%  Global Parameters
% ------------------------
tic
N    = 100;        % Population size
beta0 = 0.1;       % default selection intensity for examples

% Attacker params (attack vs H/L profitable)
cah = 0.85;    bah = 1.90;   % attacker cost/benefit vs H
cal = 0.10;    bal = 1.60;   % attacker cost/benefit vs L

% Defender params
pdh = 0.82;    % H success prob
pdl = 0.75;    % L success prob
BH  = 0.75;    BL  = 0.55;
CH  = 0.41;    CL  = 0.20;
WH  = 0.22;    WL  = 0.10;
z0  = 10;      % example number of committed H defenders

%% ============================================================
%  Pack parameters (single struct used everywhere)
% =============================================================
params_base = struct( ...
    'pdh',pdh,'pdl',pdl, ...
    'BH',BH,'BL',BL, ...
    'CH',CH,'CL',CL, ...
    'WH',WH,'WL',WL, ...
    'cah',cah,'bah',bah, ...
    'cal',cal,'bal',bal, ...
    'wA',0.5, ...           % weight attackers vs defenders
    'betaA',beta0, ...      % attacker imitation intensity
    'subsidy_on',subsidy_on ...
);

%% ============================================================
%  Quick debug: show payoffs with / without subsidy
% =============================================================
[~, PI4_debug] = stationary_block_chain(N, z0, beta0, params_base);
fprintf('Stationary distribution (z=%d, beta=%.3g): [A,H A,L NA,H NA,L] = [%.3f %.3f %.3f %.3f]\n', ...
        z0, beta0, PI4_debug);
%% ===========================
%  Helper functions
% ===========================
function pi = stationary_dist(M)
% Robust stationary distribution via power iteration.
    n = size(M,1);
    pi = ones(1,n)/n;
    tol = 1e-12;  maxit = 20000;
    for t = 1:maxit
        pi_new = pi * M;
        if norm(pi_new - pi, 1) < tol
            pi = pi_new / sum(pi_new);
            return
        end
        pi = pi_new;
    end
    pi = pi / sum(pi);
end

function [pi_block, PI4] = stationary_block_chain(N, z, betaD, params)
% Coupled attacker/defender Markov chain with z committed H defenders.
% Uses a simple subsidy model: when subsidy_on=true and z>0,
% high defenders effectively recover a fraction (z/N)*CH of their cost CH.

    pdh = params.pdh;  pdl = params.pdl;
    BH  = params.BH;   BL  = params.BL;
    CH  = params.CH;   CL  = params.CL;
    WH  = params.WH;   WL  = params.WL;
    cah = params.cah;  bah = params.bah;
    cal = params.cal;  bal = params.bal;

    if ~isfield(params,'wA'),    params.wA = 0.5;  end
    if ~isfield(params,'betaA'), params.betaA = betaD; end
    if ~isfield(params,'subsidy_on'), params.subsidy_on = false; end

    wA    = params.wA;
    wD    = 1 - wA;
    betaA = params.betaA;
    subsidy_on = params.subsidy_on;

    % --- base per-interaction payoffs ---
    fA_H  = -cah + bah*(1 - pdh);
    fA_L  = -cal + bal*(1 - pdl);
    fH_A  =  pdh*BH - CH - (1 - pdh)*WH;
    fL_A  =  pdl*BL - CL - (1 - pdl)*WL;
    fH_NA =  BH - CH;
    fL_NA =  BL - CL;

    % --- apply subsidy as effective boost to H if z>0 ---
    fH_A_eff  = fH_A;
    fH_NA_eff = fH_NA;
    if subsidy_on && z > 0
        delta = (z / N) * CH;   % average subsidy per high defender
        fH_A_eff  = fH_A  + delta;
        fH_NA_eff = fH_NA + delta;
    end

    % DEBUG (only once for a particular z,beta)
    if z == 10 && abs(betaD - 0.1) < 1e-12
        fprintf('DEBUG in stationary_block_chain: z=%d, betaD=%.3g\n', z, betaD);
        fprintf('  fH_A(base) = %.4f,  fH_A_eff = %.4f\n', fH_A, fH_A_eff);
        fprintf('  fH_NA(base)= %.4f,  fH_NA_eff= %.4f\n\n', fH_NA, fH_NA_eff);
    end

    % --- state space: (A,i) and (NA,i), i=0..N0 ---
    N0 = N - z;
    if N0 < 0, error('z cannot exceed N'); end
    nStates = 2*(N0+1);
    idxA  = @(i) (i+1);
    idxNA = @(i) (N0+1) + (i+1);

    I = []; J = []; V = [];

    for a = [1 0] % 1=A, 0=NA
        for i = 0:N0
            s = (a==1)*idxA(i) + (a==0)*idxNA(i);

            % composition incl. zealots
            yH = (i + z)/N;  
            yL = 1 - yH;

            % defender payoffs given attacker state (use eff H)
            if a==1
                piH = fH_A_eff;   piL = fL_A;
            else
                piH = fH_NA_eff;  piL = fL_NA;
            end

            % defender imitation (Fermi)
            p_LH = 1/(1+exp(-betaD*(piH - piL)));
            p_HL = 1/(1+exp(-betaD*(piL - piH)));

            if N0>0
                frac_L_learner = (N0 - i)/N0;
                frac_H_learner = i/N0;
            else
                frac_L_learner = 0; frac_H_learner = 0;
            end
            frac_H_model = (i + z)/N;
            frac_L_model = (N0 - i)/N;

            Tplus  = frac_L_learner * frac_H_model * p_LH; % i→i+1
            Tminus = frac_H_learner * frac_L_model * p_HL; % i→i-1
            TstayD = max(0, 1 - (Tplus + Tminus));

            % attacker imitation
            piA  = yH*fA_H + yL*fA_L;
            piNA = 0;
            p_AtoNA = 1/(1+exp(-betaA*(piNA - piA)));
            p_NAtoA = 1/(1+exp(-betaA*(piA  - piNA)));

            % defender moves
            if i < N0
                sp = (a==1)*idxA(i+1) + (a==0)*idxNA(i+1);
                I(end+1)=s; J(end+1)=sp; V(end+1)=wD*Tplus;
            end
            if i > 0
                sm = (a==1)*idxA(i-1) + (a==0)*idxNA(i-1);
                I(end+1)=s; J(end+1)=sm; V(end+1)=wD*Tminus;
            end

            % attacker moves
            if a==1
                sf = idxNA(i);
                I(end+1)=s; J(end+1)=sf; V(end+1)=wA*p_AtoNA;
                pStayA = 1 - p_AtoNA;
            else
                sf = idxA(i);
                I(end+1)=s; J(end+1)=sf; V(end+1)=wA*p_NAtoA;
                pStayA = 1 - p_NAtoA;
            end

            % self-loop
            I(end+1)=s; J(end+1)=s; V(end+1)=wD*TstayD + wA*pStayA;
        end
    end

    % --- transition matrix and stationary dist. ---
    M = sparse(I,J,V,nStates,nStates);
    rs = full(sum(M,2));
    fix = find(abs(rs-1)>1e-12);
    for k = fix.', M(k,k) = M(k,k) + (1 - rs(k)); end
    pi_block = stationary_dist(full(M));  % row vector

    % --- aggregate to 4 macro-states: [A,H A,L NA,H NA,L] ---
    PI4 = zeros(1,4);
    for i = 0:N0
        yH  = (i + z)/N;
        PA  = pi_block(idxA(i));
        PNA = pi_block(idxNA(i));
        PI4(1) = PI4(1) + PA  * yH;
        PI4(2) = PI4(2) + PA  * (1 - yH);
        PI4(3) = PI4(3) + PNA * yH;
        PI4(4) = PI4(4) + PNA * (1 - yH);
    end
end
%% %% %% --- Example: stationary distribution for one (z, sel) ---
z_markov  = 10;        % pick the z you want
sel_markov = 0.1;      % pick the selection intensity

[~, PI4m] = stationary_block_chain(N, z_markov, sel_markov, params);  % [A,H A,L NA,H NA,L]
pi_frac   = PI4m;
pi_pct    = 100*PI4m;

fprintf('π (%%) at z=%d, sel=%.3g: [A,H A,L NA,H NA,L] = [%.2f %.2f %.2f %.2f]\n', ...
        z_markov, sel_markov, pi_pct);

%% --- Markov diagram (macro 4-state view, illustrative edges) ---
stateLabels = {'(A,H)','(A,L)','(NA,H)','(NA,L)'};
pos = [0 1; 1 1; 0 0; 1 0];   % positions for plotting

% Example edges (you can replace wts with actual transition probs if you extract them)
src = [1 1 2 2 3 3 4 4];       % from
dst = [2 3 1 4 1 4 2 3];       % to
wts = [0.05 0.15 0.10 0.05 0.20 0.05 0.25 0.15];

G = digraph(src, dst, wts, stateLabels);
max_wt = max(wts);
lw = 1 + 10*(wts / max_wt);    % thicker edges for higher weights

figure('Name','Embedded Markov Chain — 4 macro-states','Color','w');
p = plot(G, 'XData',pos(:,1), 'YData',pos(:,2), ...
         'NodeLabel',repmat({''},1,4), ...
         'ArrowSize',12, 'LineWidth',lw, ...
         'EdgeColor',[0.2 0.2 0.8], ...
         'NodeColor',[0.1 0.1 0.1], 'MarkerSize',18);

axis equal; axis off; hold on;

% State labels
for k = 1:4
    text(pos(k,1), pos(k,2)+0.03, stateLabels{k}, ...
        'FontWeight','bold','FontSize',18, ...
        'HorizontalAlignment','center','VerticalAlignment','bottom');
end

% Node labels with π values (%)
for k = 1:4
    text(pos(k,1), pos(k,2)-0.03, sprintf('\\pi=%.3f', pi_pct(k)), ...
        'Color',[0.1 0.5 0.1],'FontWeight','bold','FontSize',18, ...
        'HorizontalAlignment','center','VerticalAlignment','top');
end

% Edge labels (transition weights, illustrative)
labeledge(p, src, dst, arrayfun(@(x) sprintf('%.3f', x), wts, 'UniformOutput', false));
%% %% --- Selection-intensity sweep (analog of beta sweep) ---
sel_vals = logspace(-2, 1, 11);   % 0.01 ... 10
PI_sel   = zeros(4, numel(sel_vals));

for k = 1:numel(sel_vals)
    s_val = sel_vals(k);
    [~, PI4k] = stationary_block_chain(N, z, s_val, params);  % z is current global z
    PI_sel(:,k) = PI4k(:);
end

figure('Name','Stationary distribution vs selection intensity \sigma','Color','w');
hold on; grid on; box on;
semilogx(sel_vals, PI_sel(1,:),'o-','LineWidth',3,'MarkerSize',5);
semilogx(sel_vals, PI_sel(2,:),'o-','LineWidth',3,'MarkerSize',5);
semilogx(sel_vals, PI_sel(3,:),'o-','LineWidth',3,'MarkerSize',5);
semilogx(sel_vals, PI_sel(4,:),'o-','LineWidth',3,'MarkerSize',5);

set(gca,'XScale','log','XTick',sel_vals);
xticklabels(arrayfun(@(v) sprintf('%.2g',v), sel_vals, 'uni',0));
ytickformat('%.2f'); ylim([0 1]);

xlabel('\beta (selection intensity)');
ylabel('Stationary probability');

legend({'\pi(A,H)','\pi(A,L)','\pi(NA,H)','\pi(NA,L)'},'Location','best');
set(gca,'FontSize',22,'FontWeight','Bold');

%% ---- 2x2 heatmaps of π_i vs (z, β) and total H share ----

Zvals_hm  = 0:10;                  % committed H defenders
beta_grid = logspace(-1, 1, 8);    % selection intensities
nZ = numel(Zvals_hm); 
nB = numel(beta_grid);

% Pack base params (including subsidy flag)
params_base = struct('pdh',pdh,'pdl',pdl, ...
                     'BH',BH,'BL',BL,'CH',CH,'CL',CL, ...
                     'WH',WH,'WL',WL, ...
                     'cah',cah,'bah',bah,'cal',cal,'bal',bal, ...
                     'wA',0.5, 'betaA',beta_grid, ...
                     'subsidy_on', subsidy_on);

PI_hm = nan(4, nZ, nB);   % [state, z, beta]

for jb = 1:nB
    b = beta_grid(jb);
    for iz = 1:nZ
        z_k = Zvals_hm(iz);
        % use a fresh params struct so we can change betaA if needed
        params = params_base;
        params.betaA = b;
        [~, PI4] = stationary_block_chain(N, z_k, b, params); % [(A,H) (A,L) (NA,H) (NA,L)]
        PI_hm(:, iz, jb) = PI4(:);
    end
end

% ---- 2x2 heatmaps with shared labels & one colorbar ----
stateNames = {'(A,H)','(A,L)','(NA,H)','(NA,L)'};
beta_ticks = [0.1 0.2 0.5 1 2 5 10];
yticks_log = log10(beta_ticks);              % because we plot log10(beta)
yticklabs  = string(beta_ticks);

fig = figure('Color','w','Name','Stationary probabilities vs (z, \beta)');
t   = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');

axs = gobjects(1,4);
for s = 1:4
    axs(s) = nexttile;
    imagesc(Zvals_hm, log10(beta_grid), squeeze(PI_hm(s,:,:))');
    axis xy; caxis([0 1]); colormap(parula);
    title(stateNames{s}, 'FontWeight','bold','FontSize',16);
    xticks(0:1:10);
    yticks(yticks_log); yticklabels([]);
    set(axs(s),'YDir','normal','FontSize',20,'LineWidth',1.1,'Box','on');
end

% Show y tick labels only on left column; x tick labels only on bottom row
set(axs([1 3]), 'YTickLabel', yticklabs);
set(axs(3:4),   'XTickLabel', string(0:1:10));

% Shared axis labels
xlabel(t,'Committed defenders  z','FontSize',28,'FontWeight','bold');
ylabel(t,'Selection intensity  \beta','FontSize',28,'FontWeight','bold');

% Shared colorbar
cb = colorbar(axs(end), 'eastoutside');
cb.Label.String   = 'Stationary probability';
cb.Label.FontSize = 28;
cb.Label.FontWeight = 'bold';
cb.Layout.Tile    = 'east';

% ----- total H share heatmap + decision boundary ----
figure('Color','w','Name','Total H-defender frequency \pi_H vs (z, \beta)');

PIH = squeeze(PI_hm(1,:,:)+PI_hm(3,:,:))';  % [beta,z]

imagesc(Zvals_hm, log10(beta_grid), PIH);
axis xy; caxis([0 1]); colormap(parula); hold on;

xlabel('Committed defenders  z','FontSize',28,'FontWeight','bold');
ylabel('Selection intensity  \beta','FontSize',28,'FontWeight','bold');

beta_ticks_l = [0.1 0.2 0.5 1 2 5 10];
yticks(log10(beta_ticks_l)); yticklabels(string(beta_ticks_l));

cb2 = colorbar('Location','eastoutside');
cb2.Label.String = 'Stationary Distribution';
cb2.Label.FontSize = 28;
cb2.Label.FontWeight = 'bold';

set(gca,'YDir','normal','FontSize',20); grid on;
title(sprintf('Frequency of H-defenders'), ...
      'FontSize',24,'FontWeight','bold');
 
%% === Welfare vs committed defenders (SW_D, SW_A, SW) ===
% Uses current N, beta, params.
% Subsidy is controlled by params.subsidy_on (true/false).

Z = 0:min(10, N-1);                   % sweep z from 0 up to 10
beta1=beta0;

% --- Recompute base (no-subsidy) payoffs from parameters ---
% Attacker vs High/Low
fA_H_base   = -cah + bah*(1 - pdh);
fA_L_base   = -cal + bal*(1 - pdl);
fNA_H_base  = 0;
fNA_L_base  = 0;

% Defender H/L vs A or NA (ordinary high defenders pay CH)
fH_A_base   = pdh*BH - CH - (1 - pdh)*WH;
fL_A_base   = pdl*BL - CL - (1 - pdl)*WL;
fH_NA_base  = BH - CH;
fL_NA_base  = BL - CL;

% Attacker welfare vector does NOT depend on subsidy
W_att_base = [fA_H_base,  fA_L_base,  fNA_H_base,  fNA_L_base];   % [(A,H) (A,L) (NA,H) (NA,L)]

% Storage
SWD = zeros(size(Z));    % defender welfare
SWA = zeros(size(Z));    % attacker welfare
SWT = zeros(size(Z));    % total welfare
piH = zeros(size(Z));    % total H share (for reference)

for k = 1:numel(Z)
    zk = Z(k);

    % ---- Effective H defender payoffs under subsidy ----
    % Same logic as in stationary_block_chain:
    % if subsidised: fH_*_eff = fH_*_base + (zk/N)*CH
    if isfield(params,'subsidy_on') && params.subsidy_on && zk > 0
        delta       = (zk / N) * CH;     % average cost offset from subsidy
        fH_A_eff    = fH_A_base  + delta;
        fH_NA_eff   = fH_NA_base + delta;
    else
        % baseline (no subsidy): ordinary payoffs
        fH_A_eff    = fH_A_base;
        fH_NA_eff   = fH_NA_base;
    end

    % Per-state defender welfare for this z
    % order: [(A,H) (A,L) (NA,H) (NA,L)]
    W_def_z = [fH_A_eff,  fL_A_base,  fH_NA_eff,  fL_NA_base];
    W_tot_z = W_def_z + W_att_base;

    % Get stationary distribution for this z
    params_z = params;
    [~, PI4] = stationary_block_chain(N, zk, beta1, params_z);  % 1x4

    % Welfare
    SWD(k) = PI4 * W_def_z.';   % defender welfare
    SWA(k) = PI4 * W_att_base.'; % attacker welfare
    SWT(k) = PI4 * W_tot_z.';   % total welfare

    % Total H share in this z (for reference)
    piH(k) = PI4(1) + PI4(3);   % (A,H)+(NA,H)
end

% ---- Plot: all three welfare curves ----
figure('Color','w','Name','Welfare vs committed defenders z');
hold on; grid on; box on;
p1 = plot(Z, SWD, 'b-o','LineWidth',4,'MarkerSize',5, ...
          'DisplayName','SW_D(z)  (defenders)');
p2 = plot(Z, SWA, 'r-s','LineWidth',4,'MarkerSize',5, ...
          'DisplayName','SW_A(z)  (attackers)');
p3 = plot(Z, SWT, 'k-^','LineWidth',4,'MarkerSize',5, ...
          'DisplayName','SW(z) = SW_D + SW_A');

% Optional: show total H share on right axis
yyaxis right
ylim([0 1]);
ylabel('\pi_H (H-defender fraction)');
yyaxis left

xlabel('Committed defenders  z');
ylabel('Welfare');
title('Defender, Attacker, and Total Welfare vs committed defenders z');
legend([p1 p2 p3], 'Location','best', 'FontSize',14);
set(gca,'FontSize',28,'FontWeight','bold');

% ---- Display computed welfare values ----
fprintf('\n=== Welfare summary (beta = %.3f, N = %d) ===\n', beta1, N);
fprintf('   z\tSW_D(z)\t\tSW_A(z)\t\tSW_T(z)\n');
fprintf('---------------------------------------------\n');
for k = 1:numel(Z)
    fprintf('%4d\t% .6f\t% .6f\t% .6f\n', Z(k), SWD(k), SWA(k), SWT(k));
end
fprintf('---------------------------------------------\n');
fprintf('Max SW_D = %.6f at z = %d\n', max(SWD), Z(SWD == max(SWD)));
fprintf('Max SW_T = %.6f at z = %d\n\n', max(SWT), Z(SWT == max(SWT)));

% ---- (Optional) defender-net welfare with a grant κ per H ----
% kappa = 0.02;
% SWD_net = SWD + kappa*Z;
% plot(Z, SWD_net, 'm-d','LineWidth',4,'MarkerSize',5, ...
%      'DisplayName', sprintf('SW_D(z) %+c \\kappa z', '+'));
% legend('Location','best');



