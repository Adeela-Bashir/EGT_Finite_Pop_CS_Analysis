%% ============================================================
%  Finite-Population Cyber Security (2-pop, 2x2) — Fermi Dynamics
%  Four monomorphic states: S1=(A,D), S2=(A,ND), S3=(NA,D), S4=(NA,ND)
%  - Computes fixation probabilities ρ_{i->j}
%  - Builds embedded 4x4 chain M (one population mutates at a time)
%  - Stationary distribution π over {S1..S4}
%  - Parameter-sweep plots
%  ------------------------------------------------------------
clear; clc;

%% -----------------------
%  Global Parameters
%  -----------------------
N    = 100;    % population size
beta = 0.1;    % selection intensity (Fermi rule; >=0)

% --- Model parameters (from your spec) ---
m  = 0;      % probability of catching attacker on unsecured system 
n  = 0;      % probability of catching attacker on secured system
u  = 0;     % attacker penalty if successful 
s  = 0;     % attacker penalty if unsuccessful 

ca = 0.41; ba = 0.90; pd = 0.26;
m = 0; u = 0; cd = 0.20; bd = 0.79; w = 0.98;

%% -----------------------
%  Helper: Side-conditional payoffs 
%  -----------------------
% Attackers' payoffs when defenders are all D or all ND
fA_D   = -ca + ba*(1 - pd) - pd*n*s - (1 - pd)*m*u;  % attacker vs all-D defenders
fNA_D  =  0;                                       % non-attacker vs D
fA_ND  = -ca + ba - m*u;                             % attacker vs all-ND defenders
fNA_ND =  0;                                       % non-attacker vs ND

% Defenders' payoffs when attackers are all A or all NA
fD_A   = -cd + pd*bd - w*(1 - pd);                 % defender vs all-A attackers
fND_A  = -w;                                       % non-defender vs A
fD_NA  = -cd + bd;                                 % defender vs NA
fND_NA =  0;                                       % non-defender vs NA

%% ============================================================
%  PART 1: Fixation probabilities for all needed one-step invasions
%          (one population mutates while the other remains fixed)
% =============================================================
% We need 8 rhos (two directions × two environments for each side).

% --- Attackers mutate (D fixed to D or ND) ---
rho_AtoNA_given_D   = rho_fermi_two_type(N, beta, fA_D,  fNA_D);   % (A -> NA) with defenders = D
rho_NAtoA_given_D   = rho_fermi_two_type(N, beta, fNA_D, fA_D);    % (NA -> A) with defenders = D

rho_AtoNA_given_ND  = rho_fermi_two_type(N, beta, fA_ND,  fNA_ND); % (A -> NA) with defenders = ND
rho_NAtoA_given_ND  = rho_fermi_two_type(N, beta, fNA_ND, fA_ND);  % (NA -> A) with defenders = ND

% --- Defenders mutate (A fixed to A or NA) ---
rho_DtoND_given_A   = rho_fermi_two_type(N, beta, fD_A,  fND_A);   % (D -> ND) with attackers = A
rho_NDtoD_given_A   = rho_fermi_two_type(N, beta, fND_A, fD_A);    % (ND -> D) with attackers = A

rho_DtoND_given_NA  = rho_fermi_two_type(N, beta, fD_NA,  fND_NA); % (D -> ND) with attackers = NA
rho_NDtoD_given_NA  = rho_fermi_two_type(N, beta, fND_NA, fD_NA);  % (ND -> D) with attackers = NA

%% Show rhos
disp('--- Fixation probabilities (Fermi, finite N) ---');
fprintf('Attackers | D fixed:   rho(A->NA|D)=%.6f, rho(NA->A|D)=%.6f\n', rho_AtoNA_given_D,  rho_NAtoA_given_D);
fprintf('Attackers | ND fixed:  rho(A->NA|ND)=%.6f, rho(NA->A|ND)=%.6f\n', rho_AtoNA_given_ND, rho_NAtoA_given_ND);
fprintf('Defenders | A fixed:   rho(D->ND|A)=%.6f, rho(ND->D|A)=%.6f\n', rho_DtoND_given_A,  rho_NDtoD_given_A);
fprintf('Defenders | NA fixed:  rho(D->ND|NA)=%.6f, rho(ND->D|NA)=%.6f\n', rho_DtoND_given_NA, rho_NDtoD_given_NA);

%% ============================================================
%  PART 2: Embedded 4x4 Markov chain M over states {S1..S4}
%          S1=(A,D), S2=(A,ND), S3=(NA,D), S4=(NA,ND)
%          Only one population mutates at a time -> scale by 1/2
% =============================================================
M = zeros(4,4);

% From S1=(A,D): attackers might switch to NA (-> S3), or defenders might switch to ND (-> S2).
M(1,3) = 0.5 * rho_AtoNA_given_D;
M(1,2) = 0.5 * rho_DtoND_given_A;

% From S2=(A,ND): attackers -> NA (-> S4), or defenders -> D (-> S1).
M(2,4) = 0.5 * rho_AtoNA_given_ND;
M(2,1) = 0.5 * rho_NDtoD_given_A;

% From S3=(NA,D): attackers -> A (-> S1), or defenders -> ND (-> S4).
M(3,1) = 0.5 * rho_NAtoA_given_D;
M(3,4) = 0.5 * rho_DtoND_given_NA;

% From S4=(NA,ND): attackers -> A (-> S2), or defenders -> D (-> S3).
M(4,2) = 0.5 * rho_NAtoA_given_ND;
M(4,3) = 0.5 * rho_NDtoD_given_NA;

% Diagonals
for r = 1:4
    M(r,r) = 1 - sum(M(r,[1:r-1 r+1:4]));
end

disp('--- Embedded Markov chain M (rows sum to 1) ---');
disp(M);

%% ============================================================
%  PART 3: Stationary distribution π over {S1..S4}
% =============================================================
pi = stationary_dist(M)*100;
disp('--- Stationary distribution π over [S1 S2 S3 S4] = [(A,D) (A,ND) (NA,D) (NA,ND)] ---');
disp(pi);

%% ===========================
%  Local Functions
% ===========================
function rho = rho_fermi_two_type(N, beta, f_resident, f_mutant)
% Fixation probability of a single mutant (type B) in a population of residents (type A)
% under pairwise comparison (Fermi rule) with your T± form:
%   T+(i) = ((N-i)/N) * (i/N) * 1/(1 + exp(-beta*(f_mut - f_res)))
%   T-(i) = (i/N) * ((N-i)/N) * 1/(1 + exp(+beta*(f_mut - f_res)))
% For constant payoffs (frequency independent), ratio r = T-(i)/T+(i) is constant:
%   r = (1 + e^{-βΔf}) / (1 + e^{+βΔf}), where Δf = f_mut - f_res
% Then:
%   ρ = (1 - r) / (1 - r^N), with the neutral limit ρ = 1/N when Δf ~ 0.

    dF = f_mutant - f_resident;
    if abs(dF) < 1e-12 || beta==0
        rho = 1.0 / N;
        return;
    end
    r = (1 + exp(-beta*dF)) / (1 + exp(+beta*dF));
    % Numerical stability
    if abs(1 - r) < 1e-14
        rho = 1.0 / N;
    else
        % Use logs if needed for large N/beta:
        % rho = (1 - r) / (1 - r^N);
        log_r = log(r);
        if log_r * N > 700   % overflow guard
            rho = 0; % r^N huge -> denominator large -> rho ~ 0
        elseif log_r * N < -700
            rho = 1; % r^N ~ 0 -> rho ~ (1 - r) / 1  (but r<<1 => rho ~1)
        else
            rho = (1 - r) / (1 - r^N);
        end
    end
end

function pi = stationary_dist(M)
% Left eigenvector for eigenvalue 1 (normalized)
    [V,D] = eig(M.');
    [~, idx] = min(abs(diag(D) - 1));
    v = V(:, idx);
    pi = (v / sum(v)).';           % row vector (sum to 1)
    pi = real(pi);                 % strip tiny imag parts
end

function [beta_g, pd_g, w_g, ca_g, cd_g, ba_g, bd_g, n_g, u_g, s_g] = ...
         apply_param(px, vx, py, vy, beta, pd, w, ca, cd, ba, bd, n, u, s)
% Apply two-parameter changes for the sweep
    beta_g = beta; pd_g = pd; w_g = w; ca_g = ca; cd_g = cd;
    ba_g = ba; bd_g = bd; n_g = n; u_g = u; s_g = s;

    beta_g = set_if_match(beta_g, 'beta', px, vx, py, vy);
    pd_g   = set_if_match(pd_g,   'pd',   px, vx, py, vy);
    w_g    = set_if_match(w_g,    'w',    px, vx, py, vy);
    ca_g   = set_if_match(ca_g,   'ca',   px, vx, py, vy);
    cd_g   = set_if_match(cd_g,   'cd',   px, vx, py, vy);
    ba_g   = set_if_match(ba_g,   'ba',   px, vx, py, vy);
    bd_g   = set_if_match(bd_g,   'bd',   px, vx, py, vy);
    n_g    = set_if_match(n_g,    'n',    px, vx, py, vy);
    u_g    = set_if_match(u_g,    'u',    px, vx, py, vy);
    s_g    = set_if_match(s_g,    's',    px, vx, py, vy);
end

function v = set_if_match(v, name, px, vx, py, vy)
    if strcmpi(px, name), v = vx; end
    if strcmpi(py, name), v = vy; end
end

function [traj, occ] = simulate_chain_M(M, state0, T)
% Simple Monte Carlo over the 4-state chain M
    K = size(M,1);
    traj = zeros(1,T); traj(1) = state0;
    for t = 2:T
        i = traj(t-1);
        p = M(i,:);
        % draw next
        r = rand;
        c = cumsum(p);
        j = find(r <= c, 1, 'first');
        traj(t) = j;
    end
    % empirical occupancy
    occ = histcounts(traj, 0.5:1:(K+0.5));
    occ = occ / sum(occ);
end
%% ============================================================
%  PART 6  One-parameter sweeps (β fixed) — Strategy evolution
%          Y-axis: stationary distribution components π_i
%          X-axis: one model parameter per subplot
%          Removed: n, s, u
% =============================================================
beta_fixed = 0.1;   % β constant for all sweeps

% Base parameter set (copied from current workspace values)
base.w  = w;   base.ca = ca; base.cd = cd; base.ba = ba; base.bd = bd;
base.pd = pd;  base.N  = N;

param_list = { ...
   'pd', linspace(0.0, 1.0, 15); ...                        % defender success prob
   'w',  linspace(0.05, 1.0, 15); ...                       % residual loss weight
   'ca', linspace(0.01, 0.95*base.w, 15); ...               % attacker cost vs H
   'cd', linspace(0.01, min(0.9*base.bd,0.9*base.w), 15); ... % defender cost vs A
   'ba', linspace(max(base.ca+0.01,0.05), 1.2, 15); ...     % attacker benefit vs H
   'bd', linspace(max(base.cd+0.01,0.05), base.w, 15) ...   % defender benefit vs A
};

figure('Name','One-parameter sweeps (stationary distribution vs parameter)','Color','w');
t=tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

print(gcf, 'Figure_5.png', '-dpng', '-r300');
% β only once at the top:
% sgtitle(t, sprintf('\\beta = %.3g', beta_fixed));

legend_labels = {'\pi_1 (A,D)','\pi_2 (A,ND)','\pi_3 (NA,D)','\pi_4 (NA,ND)'};
legend_handle = [];

for pidx = 1:size(param_list,1)
    param_name = param_list{pidx,1};
    X = param_list{pidx,2};

    % Storage for π across the sweep
    PI = zeros(4, numel(X));

    for ix = 1:numel(X)
        % Start from base, set beta=beta_fixed, change one parameter
        pars = base;
        pars.(param_name) = X(ix);

        % Enforce constraints softly
        pars = enforce_constraints_noNSU(pars);

        % Side-conditional payoffs at this grid point (n,s,u removed)
        % Attackers vs all-D and all-ND:
        fA_D_g   = -pars.ca + pars.ba*(1 - pars.pd);   % -ca + ba*(1-pd)
        fNA_D_g  =  0;
        fA_ND_g  = -pars.ca + pars.ba;                 % -ca + ba
        fNA_ND_g =  0;

        % Defenders vs all-A and all-NA:
        fD_A_g   = -pars.cd + pars.pd*pars.bd - pars.w*(1 - pars.pd);
        fND_A_g  = -pars.w;
        fD_NA_g  = -pars.cd + pars.bd;
        fND_NA_g =  0;

        % Fixations (β fixed)
        rho_AtoNA_D    = rho_fermi_two_type(pars.N, beta_fixed, fA_D_g,  fNA_D_g);
        rho_NAtoA_D    = rho_fermi_two_type(pars.N, beta_fixed, fNA_D_g, fA_D_g);
        rho_AtoNA_ND   = rho_fermi_two_type(pars.N, beta_fixed, fA_ND_g,  fNA_ND_g);
        rho_NAtoA_ND   = rho_fermi_two_type(pars.N, beta_fixed, fNA_ND_g, fA_ND_g);
        rho_DtoND_A    = rho_fermi_two_type(pars.N, beta_fixed, fD_A_g,  fND_A_g);
        rho_NDtoD_A    = rho_fermi_two_type(pars.N, beta_fixed, fND_A_g, fD_A_g);
        rho_DtoND_NA   = rho_fermi_two_type(pars.N, beta_fixed, fD_NA_g,  fND_NA_g);
        rho_NDtoD_NA   = rho_fermi_two_type(pars.N, beta_fixed, fND_NA_g, fD_NA_g);

        % Embedded 4x4 chain
        M_g = zeros(4,4);
        M_g(1,3) = 0.5 * rho_AtoNA_D;    M_g(1,2) = 0.5 * rho_DtoND_A;
        M_g(2,4) = 0.5 * rho_AtoNA_ND;   M_g(2,1) = 0.5 * rho_NDtoD_A;
        M_g(3,1) = 0.5 * rho_NAtoA_D;    M_g(3,4) = 0.5 * rho_DtoND_NA;
        M_g(4,2) = 0.5 * rho_NAtoA_ND;   M_g(4,3) = 0.5 * rho_NDtoD_NA;
        for r = 1:4, M_g(r,r) = 1 - sum(M_g(r,[1:r-1 r+1:4])); end

        PI(:,ix) = stationary_dist(M_g).';   % store as column
    end

    % ---- Plot this parameter’s sweep (discrete points) ----
nexttile; hold on; grid on;

% one marker style per state
mk = {'o-','s-','^-','d-'};
C  = lines(4);
h  = gobjects(1,4);

for k = 1:4
    h(k) = plot(X, PI(k,:), mk{k}, ...
        'LineWidth', 2.2, ...
        'MarkerSize', 6, ...
        'MarkerFaceColor', C(k,:), ...
        'Color', C(k,:));           % markers at each sampled X (discrete)
end

ylim([0 1]);
xlim([min(X) max(X)]);
set(gca,'XTick',X);                 % show the discrete sample points on x-axis
xlabel(param_name,'Interpreter','none');
ylabel('Stationary probability');
set(gca,'FontSize',16,'FontWeight','Bold');
xtickformat('%.2f');
ytickformat('%.2f');
end
if isempty(legend_handle) || ~isvalid(legend_handle)
    legend_handle = legend(h, legend_labels, 'Location','best');
end

%% ---------------------------
%  Local helper: constraints (no n,s,u)
% ---------------------------
function pars = enforce_constraints_noNSU(pars)
% Enforce:
%  0 < w <= 1
%  0 < ca < w
%  0 < cd < bd <= w
%  ca < ba
%  0 <= pd <= 1
    epsv = 1e-6;

    pars.w  = min(max(pars.w,  0.01), 1.0);
    pars.pd = min(max(pars.pd, 0.0),  1.0);

    % bd ≤ w
    if pars.bd > pars.w
        pars.bd = pars.w;
    end
    % cd < bd
    pars.cd = min(pars.cd, pars.bd - epsv);
    pars.cd = max(pars.cd, 0.0 + epsv);

    % ca < w
    pars.ca = min(pars.ca, pars.w - epsv);
    pars.ca = max(pars.ca, 0.0 + epsv);

    % ba > ca
    if pars.ba <= pars.ca + epsv
        pars.ba = pars.ca + 2*epsv;
    end
end
%% ============================================================
%  PART 7 (edited): Markov diagram — square layout, clean labels
%  Assumes M (4x4) and optional pi are in workspace
% =============================================================

% State names (row/col order)
stateLabels = {'(A,D)','(A,ND)','(NA,D)','(NA,ND)'};

% Build edge list (exclude self-loops and tiny weights)
src = []; dst = []; wts = [];
for i = 1:4
    for j = 1:4
        if i~=j && M(i,j) > 1e-6
            src(end+1) = i; %#ok<AGROW>
            dst(end+1) = j; %#ok<AGROW>
            wts(end+1) = M(i,j); %#ok<AGROW>
        end
    end
end
G = digraph(src, dst, wts, stateLabels);

% Square coordinates: S1 top-left, S2 top-right, S3 bottom-left, S4 bottom-right
pos = [0 1; 1 1; 0 0; 1 0];   % rows map to nodes 1..4

% Edge widths scaled for visibility (guard against division by zero)
if isempty(wts)
    lw = 1;
else
    maxw = max(wts);
    if maxw == 0, maxw = 1; end
    lw = 0.5 + 8 * (wts / maxw);
end

figure('Name','Embedded Markov Chain over Strategy Pairs','Color','w');
% exportgraphics(figure, 'figure_2_b.png', 'Resolution', 300);

p = plot(G, ...
    'XData', pos(:,1), 'YData', pos(:,2), ...
    'NodeLabel', repmat({''},1,4), ...  % suppress default labels
    'ArrowSize', 12, ...
    'LineWidth', lw, ...
    'EdgeColor', [0.15 0.15 0.7], ...
    'NodeColor', [0.1 0.1 0.1], ...
    'MarkerSize', 18);

% title('Strategy-Pair Markov Chain');
axis equal; axis off; hold on;

% Draw node circles explicitly (white fill, black edge)
scatter(pos(:,1), pos(:,2), 360, 'o', ...
    'MarkerEdgeColor',[0 0 0], 'MarkerFaceColor',[1 1 1], 'LineWidth',1.6);

% --- Single, bold state labels (closer to nodes) ---
labelOffset = 0.03;     % smaller => closer to the node
for k = 1:4
    text(pos(k,1), pos(k,2)+labelOffset, stateLabels{k}, ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
        'FontWeight','bold', 'FontSize',20);
end

% --- Optional: π values (closer to nodes) ---
if exist('pi','var')
    piOffset = 0.03;   % smaller => closer to the node
    for k = 1:4
        text(pos(k,1), pos(k,2)-piOffset, sprintf('\\pi=%.3f', pi(k)), ...
            'HorizontalAlignment','center', 'VerticalAlignment','top', ...
            'Color',[0.1 0.5 0.1], 'FontWeight','bold', 'FontSize',20);
    end
end

% --- Custom edge labels placed farther from the edges ---
edgeLabelOffset = 0.08;   % larger => farther from edge (reduces overlap)
for e = 1:numel(wts)
    i = src(e); j = dst(e);
    xi = pos(i,1); yi = pos(i,2);
    xj = pos(j,1); yj = pos(j,2);

    % Midpoint of the edge
    mx = (xi + xj)/2; my = (yi + yj)/2;

    % Edge direction and a perpendicular unit normal
    dx = xj - xi; dy = yj - yi;
    L  = hypot(dx, dy); if L == 0, L = 1; end
    nx = -dy / L; ny = dx / L;   % rotate by +90°

    % Place label away from edge along the normal
    lx = mx + edgeLabelOffset * nx;
    ly = my + edgeLabelOffset * ny;

    text(lx, ly, sprintf('%.3f', wts(e)), ...
        'HorizontalAlignment','center', 'BackgroundColor','w', ...
        'Margin', 0.1, 'EdgeColor','none', 'FontSize',20);
end

%% %% ============================================================
%  PART 8: Beta sweep — Stationary distribution vs selection (β)
%          X-axis: beta in [0.001, 1] (log scale)
%          Y-axis: stationary probabilities π_i
% =============================================================

% Use current workspace parameters: N, w, ca, cd, ba, bd, pd
% Assumes helper functions rho_fermi_two_type() and stationary_dist() exist.

beta_vals = logspace(-3, 0, 200);   % 0.001 -> 1
PI_beta   = zeros(4, numel(beta_vals));

% Side-conditional payoffs (no n,s,u case)
fA_D   = -ca + ba*(1 - pd);   % attacker vs all-D defenders
fNA_D  =  0;
fA_ND  = -ca + ba;            % attacker vs all-ND defenders
fNA_ND =  0;

fD_A   = -cd + pd*bd - w*(1 - pd);  % defender vs all-A attackers
fND_A  = -w;
fD_NA  = -cd + bd;                  % defender vs all-NA attackers
fND_NA =  0;

for ib = 1:numel(beta_vals)
    b = beta_vals(ib);

    % Fixations (attackers mutate with defenders fixed; defenders mutate with attackers fixed)
    rho_AtoNA_D    = rho_fermi_two_type(N, b, fA_D,  fNA_D);
    rho_NAtoA_D    = rho_fermi_two_type(N, b, fNA_D, fA_D);
    rho_AtoNA_ND   = rho_fermi_two_type(N, b, fA_ND,  fNA_ND);
    rho_NAtoA_ND   = rho_fermi_two_type(N, b, fNA_ND, fA_ND);

    rho_DtoND_A    = rho_fermi_two_type(N, b, fD_A,  fND_A);
    rho_NDtoD_A    = rho_fermi_two_type(N, b, fND_A, fD_A);
    rho_DtoND_NA   = rho_fermi_two_type(N, b, fD_NA,  fND_NA);
    rho_NDtoD_NA   = rho_fermi_two_type(N, b, fND_NA, fD_NA);

    % Embedded 4x4 chain M for this beta
    M_b = zeros(4,4);
    % S1=(A,D)
    M_b(1,3) = 0.5 * rho_AtoNA_D;
    M_b(1,2) = 0.5 * rho_DtoND_A;
    % S2=(A,ND)
    M_b(2,4) = 0.5 * rho_AtoNA_ND;
    M_b(2,1) = 0.5 * rho_NDtoD_A;
    % S3=(NA,D)
    M_b(3,1) = 0.5 * rho_NAtoA_D;
    M_b(3,4) = 0.5 * rho_DtoND_NA;
    % S4=(NA,ND)
    M_b(4,2) = 0.5 * rho_NAtoA_ND;
    M_b(4,3) = 0.5 * rho_NDtoD_NA;

    for r = 1:4, M_b(r,r) = 1 - sum(M_b(r,[1:r-1 r+1:4])); end

    PI_beta(:, ib) = stationary_dist(M_b).';  % store column
end

% Plot π_i vs beta
figure('Name','Stationary distribution vs selection intensity \beta','Color','w');
hold on;
h1 = plot(beta_vals, PI_beta(1,:), '-',  'LineWidth', 5);   % π1: (A,D)
h2 = plot(beta_vals, PI_beta(2,:), '-', 'LineWidth', 5);   % π2: (A,ND)
h3 = plot(beta_vals, PI_beta(3,:), '-',  'LineWidth', 5);   % π3: (NA,D)
h4 = plot(beta_vals, PI_beta(4,:), '-', 'LineWidth', 5);   % π4: (NA,ND)
set(gca, 'XScale','log'); grid on; ylim([0 1]);
xlabel('Selection Intensity (\beta)');
ylabel('Stationary Distribution');
% title(sprintf('N=%d, w=%.2f, ca=%.2f, cd=%.2f, ba=%.2f, bd=%.2f, pd=%.2f', N,w,ca,cd,ba,bd,pd));
legend([h1 h2 h3 h4], {'\pi_1 (A,D)','\pi_2 (A,ND)','\pi_3 (NA,D)','\pi_4 (NA,ND)'}, 'FontSize', 20, 'Location','best');
set(gca, 'FontSize', 36, 'FontWeight', 'Bold');

%% %% ===== Robustness of stationary distribution π via random games (line + std band) =====

% tic
beta_fixed     = 0.1;        % fixed selection intensity
MC_per_value   = 10000;      % random games per grid point

% Base parameter set (from current workspace)
base.w  = w;   base.ca = ca; base.cd = cd; base.ba = ba; base.bd = bd;
base.pd = pd;  base.N  = N;

% Same parameter ranges as your original sweep
param_list = { ...
   'pd', linspace(0.0, 1.0, 11); ...      % defender success prob
   'w',  linspace(0.05, 1.0, 11); ...     % residual loss weight
   'ca', linspace(0.0, 1.0, 11); ...      % attacker cost vs H
   'cd', linspace(0.0, 1.0, 11); ... % defender cost vs A
   'ba', linspace(0, 1.2, 11); ...     % attacker benefit vs H
   'bd', linspace(0, 1.0, 11) ...   % defender benefit vs A
};

figure('Name','Random-games robustness (π vs parameter, mean ± std)','Color','w');
t = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');
title(t, sprintf('Frequency of strategies (\\beta = %.3g)', beta_fixed), ...
      'FontSize',18,'FontWeight','bold');

legend_labels = {'\pi_1 (A,D)','\pi_2 (A,ND)','\pi_3 (NA,D)','\pi_4 (NA,ND)'};
C  = lines(4);                 % colours for the 4 states
mk = {'o','s','^','d'};        % marker shapes
legend_handle = [];

for pidx = 1:size(param_list,1)
    param_name = param_list{pidx,1};
    X          = param_list{pidx,2};
    nX         = numel(X);

    % mean & std over random games for each π_k and each grid point X(i)
    PI_mean = zeros(4, nX);
    PI_std  = zeros(4, nX);

    for ix = 1:nX
        xval = X(ix);

        % collect π samples for this grid value
        PI_samples = nan(4, MC_per_value);

        for rep = 1:MC_per_value
            % start from base and fix the swept parameter
            pars = base;
            pars.(param_name) = xval;

            % randomly sample the OTHER parameters within their global ranges
            for qidx = 1:size(param_list,1)
                other = param_list{qidx,1};
                if strcmp(other, param_name), continue; end
                range_q = param_list{qidx,2};
                qmin = min(range_q); qmax = max(range_q);
                pars.(other) = qmin + (qmax-qmin)*rand;
            end

            % enforce constraints
            pars = enforce_constraints_noNSU(pars);

            % ---- payoff definitions (same as your original code) ----
            % Attackers vs all-D / all-ND:
            fA_D_g   = -pars.ca + pars.ba*(1 - pars.pd);
            fNA_D_g  =  0;
            fA_ND_g  = -pars.ca + pars.ba;
            fNA_ND_g =  0;

            % Defenders vs all-A / all-NA:
            fD_A_g   = -pars.cd + pars.pd*pars.bd - pars.w*(1 - pars.pd);
            fND_A_g  = -pars.w;
            fD_NA_g  = -pars.cd + pars.bd;
            fND_NA_g =  0;

            % fixation probabilities (β fixed)
            rho_AtoNA_D  = rho_fermi_two_type(pars.N, beta_fixed, fA_D_g,  fNA_D_g);
            rho_NAtoA_D  = rho_fermi_two_type(pars.N, beta_fixed, fNA_D_g, fA_D_g);
            rho_AtoNA_ND = rho_fermi_two_type(pars.N, beta_fixed, fA_ND_g,  fNA_ND_g);
            rho_NAtoA_ND = rho_fermi_two_type(pars.N, beta_fixed, fNA_ND_g, fA_ND_g);

            rho_DtoND_A  = rho_fermi_two_type(pars.N, beta_fixed, fD_A_g,  fND_A_g);
            rho_NDtoD_A  = rho_fermi_two_type(pars.N, beta_fixed, fND_A_g, fD_A_g);
            rho_DtoND_NA = rho_fermi_two_type(pars.N, beta_fixed, fD_NA_g,  fND_NA_g);
            rho_NDtoD_NA = rho_fermi_two_type(pars.N, beta_fixed, fND_NA_g, fD_NA_g);

            % embedded 4×4 Markov chain
            M_g = zeros(4,4);
            M_g(1,3) = 0.5 * rho_AtoNA_D;    M_g(1,2) = 0.5 * rho_DtoND_A;
            M_g(2,4) = 0.5 * rho_AtoNA_ND;   M_g(2,1) = 0.5 * rho_NDtoD_A;
            M_g(3,1) = 0.5 * rho_NAtoA_D;    M_g(3,4) = 0.5 * rho_DtoND_NA;
            M_g(4,2) = 0.5 * rho_NAtoA_ND;   M_g(4,3) = 0.5 * rho_NDtoD_NA;
            for r = 1:4
                M_g(r,r) = 1 - sum(M_g(r,[1:r-1 r+1:4]));
            end

            % stationary distribution for this random game
            PI_samples(:,rep) = stationary_dist(M_g).';
        end

        % mean & std across random games at X(ix)
        PI_mean(:,ix) = mean(PI_samples, 2, 'omitnan');
        PI_std(:,ix)  = std( PI_samples, 0, 2, 'omitnan');
    end

    % ---- plotting: line + shaded std band for each π_k ----
    ax = nexttile; hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    h_local = gobjects(1,4);
    for k = 1:4
        mu = PI_mean(k,:);
        sd = PI_std(k,:);
        % clamp to [0,1]
        y_lo = max(0, mu - sd);
        y_hi = min(1, mu + sd);

        % shaded band
        fill(ax, [X fliplr(X)], [y_lo fliplr(y_hi)], ...
             C(k,:), 'FaceAlpha',0.15, 'EdgeColor','none');

        % mean line
        h_local(k) = plot(ax, X, mu, ...
            'LineWidth', 2.2, ...
            'Color', C(k,:), ...
            'Marker', mk{k}, ...
            'MarkerSize', 6, ...
            'MarkerFaceColor', C(k,:));
    end

    ylim(ax,[0 1]);
    xlim(ax,[min(X) max(X)]);
    set(ax, 'XTick',X, 'FontSize',20,'FontWeight','Bold');
    xtickformat(ax,'%.2f');
    yticks(ax,0:0.25:1);          % exactly 5 ticks: 0,0.25,0.5,0.75,1
    ytickformat(ax,'%.2f');

    xlabel(ax, param_name, 'Interpreter','none');
    % no per-subplot y-label here
    % ylabel(ax, 'Stationary probability');

    % grab handles for legend from first panel
    if isempty(legend_handle)
        legend_handle = legend(ax, h_local, legend_labels, ...
                               'Location','bestoutside', 'FontSize',11);
    end
end

% single shared y-label for the whole tiled layout
ylabel(t, 'Stationary probability', 'FontSize',22,'FontWeight','bold');

toc



