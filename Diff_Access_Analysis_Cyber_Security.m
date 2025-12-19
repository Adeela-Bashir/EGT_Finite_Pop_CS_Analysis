%% ============================================================
%  Differential-Access Defence — Finite Population (Fermi)
%  States: S1=(A,H), S2=(A,L), S3=(NA,H), S4=(NA,L)
%  - Per-interaction payoffs
%  - Average payoffs Π (with zealots z only in Π_A)
%  - Fixation probabilities for one-population mutations
%  - 4x4 transition matrix M and stationary distribution π
%  - Markov diagram
%  - Parameter sweeps (π vs parameters) and β-sweep
% ------------------------------------------------------------
clear; clc; close all;
rng(7);
FS = 18; LW = 2.6;

%% -----------------------
%  Global Parameters
% ------------------------
tic
N    = 100;        % Population size
beta = 0.1;        % selection intensity

% Attacker params (make attack vs H/L profitable)
cah = 0.85;    bah = 1.90;   % attacker cost/benefit vs H
cal = 0.1;    bal = 1.60;   % attacker cost/benefit vs L
 
% % Defender params
pdh = 0.82;    % H success prob (still high)
pdl = 0.75;    % L success prob
BH  = 0.75;    BL  = 0.55;
CH  = 0.41;    CL  = 0.20;
WH  = 0.22;    WL  = 0.10;
z   = 0;       % no zealots for baseline

%% ============================================================
%  1) Per-interaction payoffs 
%     Rows: attacker (A, NA), Cols: defender (H, L)
% =============================================================
% Attacker vs High/Low
fA_H   = -cah + bah*(1 - pdh);
fA_L   = -cal + bal*(1 - pdl);
fNA_H  = 0; fNA_L  = 0;

% Defender H/L vs A or NA
fH_A   = pdh*BH - CH - (1 - pdh)*WH;
fL_A   = pdl*BL - CL - (1 - pdl)*WL;
fH_NA  = BH - CH;
fL_NA  = BL - CL;

disp('Per-interaction payoffs (corrected):');
disp(table(fA_H,fA_L,fNA_H,fNA_L,fH_A,fL_A,fH_NA,fL_NA, ...
    'VariableNames',{'fA_H','fA_L','fNA_H','fNA_L','fH_A','fL_A','fH_NA','fL_NA'}));
disp(fA_H+fH_A);
disp(fA_L+fL_A);
disp(fH_NA+fNA_H);
disp(fL_NA+fNA_L);
%% ============================================================
%  1b) Average payoffs with zealots in Π_A
%      mA = number of attackers choosing A (0..N for convenience)
%      mH = number of ordinary H among defenders (0..N-z)
% =============================================================
% Define function handles for later reuse / inspection:
PiH_fun  = @(mA) ( mA .* fH_A  + (N - mA) .* fH_NA ) / N;
PiL_fun  = @(mA) ( mA .* fL_A  + (N - mA) .* fL_NA ) / N;
PiA_fun  = @(mH) ( (mH + z) .* fA_H + (N - mH - z) .* fA_L ) / N;
PiNA_fun = @(mH) ( (mH)    .* fNA_H + (N - mH)     .* fNA_L ) / (N); % = 0 

%% ============================================================
%  2) Fixation probabilities needed for the embedded 4-state chain
%     (Only one population mutates at a time; Fermi pairwise comparison)
% =============================================================
% Attackers mutate with defenders fixed (H or L)
rho_AtoNA_given_H = rho_fermi_two_type(N, beta, fA_H,  fNA_H);
rho_NAtoA_given_H = rho_fermi_two_type(N, beta, fNA_H, fA_H);
rho_AtoNA_given_L = rho_fermi_two_type(N, beta, fA_L,  fNA_L);
rho_NAtoA_given_L = rho_fermi_two_type(N, beta, fNA_L, fA_L);

% Defenders mutate H <-> L with attackers fixed (A or NA) — zealot-aware, no payoff changes
Neff = N - z;
if Neff >= 2
    rho_HtoL_given_A  = rho_fixation_with_zealots(N, z, beta, fH_A,  fL_A);
    rho_LtoH_given_A  = rho_fixation_with_zealots(N, z, beta, fL_A,  fH_A);
    rho_HtoL_given_NA = rho_fixation_with_zealots(N, z, beta, fH_NA, fL_NA);
    rho_LtoH_given_NA = rho_fixation_with_zealots(N, z, beta, fL_NA, fH_NA);
else
    rho_HtoL_given_A  = 0;  rho_LtoH_given_A  = 0;
    rho_HtoL_given_NA = 0;  rho_LtoH_given_NA = 0;
end

disp('Fixation probabilities (ρ):');
disp(table(rho_AtoNA_given_H, rho_NAtoA_given_H, rho_AtoNA_given_L, rho_NAtoA_given_L, ...
           rho_HtoL_given_A,  rho_LtoH_given_A,  rho_HtoL_given_NA, rho_LtoH_given_NA));
%% % --- Stationary distribution ---
params = struct('pdh',pdh,'pdl',pdl,'BH',BH,'BL',BL,'CH',CH,'CL',CL, ...
                'WH',WH,'WL',WL,'cah',cah,'bah',bah,'cal',cal,'bal',bal);
params.dFD = 0.3;   % defender externality: positive makes H relatively better as yH grows
params.gA  = 0.5;   % attacker deterrence: positive makes A less profitable as yH grows

[~, PI4] = stationary_block_chain(N, z, beta, params);   % row: [(A,H) (A,L) (NA,H) (NA,L)]
pi_frac  = PI4;                  % fractions in [0,1]
pi_pct   = 100*PI4;              % percent for printing/labels

fprintf('π (%%) = [A,H A,L NA,H NA,L] = [%.3f %.3f %.3f %.3f]\n', pi_pct);

%% --- Social welfare from payoffs × stationary distribution ---
SW_states = [ (fA_H + fH_A)*pi_frac(1), ...
              (fA_L + fL_A)*pi_frac(2), ...
              (fNA_H + fH_NA)*pi_frac(3), ...
              (fNA_L + fL_NA)*pi_frac(4) ];

W_total = sum(SW_states);

disp('Social welfare contribution per state:');
disp(table({'(A,H)';'(A,L)';'(NA,H)';'(NA,L)'}, SW_states(:), ...
    'VariableNames', {'State','SW_State'}));

fprintf('Total social welfare (Σ payoff × π_i): %.4f\n', W_total);
%% --- Social welfare of defender and attacker separately ---
SW_D_states = [ (fH_A)*pi_frac(1), ...
              (fL_A)*pi_frac(2), ...
              (fH_NA)*pi_frac(3), ...
              (fL_NA)*pi_frac(4) ];
SW_A_states = [ (fA_H)*pi_frac(1), ...
              (fA_L)*pi_frac(2), ...
              (fNA_H)*pi_frac(3), ...
              (fNA_L)*pi_frac(4) ];

W_D_total = sum(SW_D_states);
W_A_total = sum(SW_A_states);

fprintf('\nSocial welfare of defender: %.4f\n', W_D_total);
fprintf('Social welfare of attacker: %.4f', W_A_total);

%% % --- Markov Diagram ---
stateLabels = {'(A,H)','(A,L)','(NA,H)','(NA,L)'};
pos = [0 1; 1 1; 0 0; 1 0];

% --- Define directed edges and their weights (transition strengths) ---
% You can replace these 0.5s with actual transition values from your matrix M if available.
src = [1 1 2 2 3 3 4 4];      % from
dst = [2 3 1 4 1 4 2 3];      % to
wts = [0.05 0.15 0.10 0.05 0.20 0.05 0.25 0.15];   % example edge probabilities

% --- Build graph ---
G = digraph(src, dst, wts, stateLabels);

% --- Scale edge width based on weights ---
max_wt = max(wts);
lw = 1 + 10*(wts / max_wt);   % thicker for higher transition probability

figure('Name','Embedded Markov Chain — block chain π','Color','w');
p = plot(G, 'XData',pos(:,1), 'YData',pos(:,2), ...
         'NodeLabel',repmat({''},1,4), ...
         'ArrowSize',12, 'LineWidth',lw, ...
         'EdgeColor',[0.2 0.2 0.8], ...
         'NodeColor',[0.1 0.1 0.1], 'MarkerSize',18);
axis equal; axis off; hold on;

% --- State labels ---
for k = 1:4
    text(pos(k,1), pos(k,2)+0.03, stateLabels{k}, ...
        'FontWeight','bold','FontSize',18, ...
        'HorizontalAlignment','center','VerticalAlignment','bottom');
end

% --- Node labels with π values (in %) ---
for k = 1:4
    text(pos(k,1), pos(k,2)-0.03, sprintf('\\pi=%.3f', pi_pct(k)), ...
        'Color',[0.1 0.5 0.1],'FontWeight','bold','FontSize',18, ...
        'HorizontalAlignment','center','VerticalAlignment','top');
end

% --- Add edge labels (transition probabilities) ---
labeledge(p, src, dst, arrayfun(@(x) sprintf('%.3f', x), wts, 'UniformOutput', false));

%% -------- local function: uses SAME stationary_block_chain -------
function SWz = local_SW_with_funded_zealots(N, z, beta, params, ...
                                            fH_A, fL_A, fH_NA, fL_NA, ...
                                            fA_H, fA_L, fH_A_noCH, fH_NA_noCH)
    [pi_block, ~] = stationary_block_chain(N, z, beta, params);  % same model
    N0   = N - z;
    idxA  = @(i) (i+1);
    idxNA = @(i) (N0+1) + (i+1);

    SWz = 0;
    for i = 0:N0
        yH      = (i + z)/N;                 % total H share
        fracH_o =  i       /N;               % ordinary H fraction
        fracH_z =  z       /N;               % zealot H fraction
        fracL   = (N - i - z)/N;             % L fraction

        % Defender welfare (zealots use no-CH versions), Attacker welfare as usual
        W_A  = (fracH_o*fH_A  + fracH_z*fH_A_noCH + fracL*fL_A) ...
             + (yH*fA_H + (1-yH)*fA_L);
        W_NA = (fracH_o*fH_NA + fracH_z*fH_NA_noCH + fracL*fL_NA) ...
             + 0;

        SWz = SWz + pi_block(idxA(i))*W_A + pi_block(idxNA(i))*W_NA;
    end
end

%% Beta sweep
beta_pts = logspace(-2, 1, 11);
PI_beta  = zeros(4, numel(beta_pts));
for k = 1:numel(beta_pts)
    b = beta_pts(k);
    [~, PI4k] = stationary_block_chain(N, z, b, params);
    PI_beta(:,k) = PI4k(:);
end

figure('Name','Stationary distribution vs \\beta (block chain, discrete)','Color','w');
hold on; grid on; box on;
semilogx(beta_pts, PI_beta(1,:),'o-','LineWidth',3,'MarkerSize',5);
semilogx(beta_pts, PI_beta(2,:),'o-','LineWidth',3,'MarkerSize',5);
semilogx(beta_pts, PI_beta(3,:),'o-','LineWidth',3,'MarkerSize',5);
semilogx(beta_pts, PI_beta(4,:),'o-','LineWidth',3,'MarkerSize',5);
set(gca,'XScale','log','XTick',beta_pts);
xticklabels(arrayfun(@(v) sprintf('%.2g',v), beta_pts, 'uni',0));
ytickformat('%.2f'); ylim([0 1]);
xlabel('\beta'); ylabel('Stationary probability');
legend({'\pi(A,H)','\pi(A,L)','\pi(NA,H)','\pi(NA,L)'},'Location','best');
set(gca,'FontSize',22,'FontWeight','Bold');
%% %% ============================================================
%  5) Parameter sweeps (π vs parameter) — DISCRETE POINTS
% ============================================================
figure('Name','Sensitivity (π vs parameter) — block chain (discrete)','Color','w');
tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

z_param = z;   % commitment level held fixed during sweeps
base = struct('pdh',pdh,'pdl',pdl,'BH',BH,'BL',BL,'CH',CH,'CL',CL, ...
              'WH',WH,'WL',WL,'cah',cah,'bah',bah,'cal',cal,'bal',bal);

% Choose small discrete grids (adjust as you like)
sweep_list = { ...
  'pdh', linspace(0.00, 1.00, 11); ...
  'pdl', linspace(0.00, 1.00, 11); ...
  'BH',  linspace(0.10, 1.20, 11); ...
  'BL',  linspace(0.05, 1.00, 11); ...
  'CH',  linspace(0.01, 0.90, 11); ...
  'CL',  linspace(0.00, 0.50, 11) ...
};

for p = 1:size(sweep_list,1)
    pname = sweep_list{p,1};
    X     = sweep_list{p,2};                 % discrete vector
    PI    = zeros(4, numel(X));

    for i = 1:numel(X)
        prm = base; prm.(pname) = X(i);
        [~, PI4] = stationary_block_chain(N, z_param, beta, prm);
        PI(:,i) = PI4(:);
    end

    nexttile; hold on; grid on; box on;
    % markers + lines so the discreteness is visible
    h = plot(X, PI(1,:), 'o-', X, PI(2,:), 'o-', X, PI(3,:), 'o-', X, PI(4,:), 'o-', ...
             'LineWidth', 2.4, 'MarkerSize', 5, 'MarkerFaceColor','auto');
    xlim([min(X) max(X)]); ylim([0 1]);
    set(gca,'XTick',X);              % show only the sampled points
    xtickformat('%.2f'); ytickformat('%.2f');
    xlabel(pname,'Interpreter','none'); ylabel('Stationary probability');
    set(gca,'FontSize',16,'FontWeight','Bold');
    if p==1
        legend(h, {'\pi(A,H)','\pi(A,L)','\pi(NA,H)','\pi(NA,L)'}, 'Location','best');
    end
end
%% %% ===========================
%  Helpers
% ===========================
function rho = rho_fermi_two_type(N, beta, f_resident, f_mutant)
% Fixation probability of one mutant (mutant fitness f_mutant) in size-N residents.
    dF = f_mutant - f_resident;
    if beta==0 || abs(dF) < 1e-14, rho = 1.0/N; return; end
    r = (1 + exp(-beta*dF)) / (1 + exp(+beta*dF));
    if abs(1 - r) < 1e-14, rho = 1.0/N; return; end
    log_r = log(r);
    if log_r*N > 700,  rho = 0;
    elseif log_r*N < -700, rho = 1;
    else, rho = (1 - r) / (1 - r^N);
    end
end

function pi = stationary_dist(M)
% Robust stationary distribution via power iteration (works when M ~ absorbing).
    n = size(M,1);
    pi = ones(1,n)/n;              % start uniform
    tol = 1e-12;  maxit = 20000;
    for t = 1:maxit
        pi_new = pi * M;
        if norm(pi_new - pi, 1) < tol
            pi = pi_new / sum(pi_new);
            return
        end
        pi = pi_new;
    end
    pi = pi / sum(pi);             % fallback
end


function rho = rho_fixation_with_zealots(N, z, beta, piH, piL)
    N0 = N - z; if N0 < 2, rho = 0; return; end
    p_LH = 1/(1 + exp(-beta*(piH - piL)));   % L imitates H
    p_HL = 1/(1 + exp(-beta*(piL - piH)));   % H imitates L
    Tplus  = @(i) ((N0 - i)/N0) .* ((i + z)/N) .* p_LH;   % i -> i+1
    Tminus = @(i) ( i / N0)    .* ((N0 - i)/N) .* p_HL;   % i -> i-1
    kmax = N0 - 1; if kmax < 1, rho = 0; return; end
    prodcum = 1; sumterm = 0;
    for k = 1:kmax
        prodcum = prodcum * (Tminus(k)/Tplus(k));
        sumterm = sumterm + prodcum;
    end
    rho = 1/(1 + sumterm);
end
%% % --- One-panel sweep over z for beta = 0.1, with A- and NA- totals added ---
Zvals   = 0:20;
b       = 0.1;                                % <-- only beta = 0.1
colors  = lines(4);
stateNames = {'\pi_1 (A,H)','\pi_2 (A,L)','\pi_3 (NA,H)','\pi_4 (NA,L)'};

PIz = nan(4, numel(Zvals));
for iz = 1:numel(Zvals)
    z_k = Zvals(iz);
    [~, PI4] = stationary_block_chain(N, z_k, b, params);  % [(A,H) (A,L) (NA,H) (NA,L)]
    PIz(:, iz) = PI4(:);
end

figure('Color','w','Name','Stationary distribution vs z (β=0.1)'); hold on; grid on; box on;
plot(Zvals, PIz(1,:), '-',  'LineWidth', 4, 'Color', colors(1,:)); % (A,H)
plot(Zvals, PIz(2,:), '-',  'LineWidth', 4, 'Color', colors(2,:)); % (A,L)
plot(Zvals, PIz(3,:), '-',  'LineWidth', 4, 'Color', colors(3,:)); % (NA,H)
plot(Zvals, PIz(4,:), '-',  'LineWidth', 4, 'Color', colors(4,:)); % (NA,L)

% Totals
piH_total  = PIz(1,:) + PIz(3,:);                      % total H defenders
piA_total  = PIz(1,:) + PIz(2,:);                      % total Attackers
piNA_total = PIz(3,:) + PIz(4,:);                      % total Non-Attackers

% Overlays (use distinct linestyles)
plot(Zvals, piH_total,  'k--', 'LineWidth', 3);        % π_H
plot(Zvals, piA_total,  'm-.', 'LineWidth', 2.8);      % A = (A,H)+(A,L)
plot(Zvals, piNA_total, 'c:',  'LineWidth', 3.2);      % NA = (NA,H)+(NA,L)

xlabel('z (committed H defenders)'); ylabel('Stationary probability');
xlim([Zvals(1) Zvals(end)]); ylim([0 1]);
title('\beta = 0.1');
legend({'\pi_1 (A,H)','\pi_2 (A,L)','\pi_3 (NA,H)','\pi_4 (NA,L)', ...
        '\pi_H=\pi_1+\pi_3', 'A total = \pi_1+\pi_2', 'NA total = \pi_3+\pi_4'}, ...
       'Location','best', 'FontSize',14); legend boxon;
set(gca,'FontSize',28, 'FontWeight','Bold');

%% ============================================================
%  Effect of committed defenders z
%  Four panels for β = {0.001, 0.1, 1, 10}
% ============================================================
Zvals  = 0:20;
betas4 = [1e-3, 0.1, 1, 10];
stateNames = {'\pi_1 (A,H)','\pi_2 (A,L)','\pi_3 (NA,H)','\pi_4 (NA,L)'};
colors = lines(4);

% pack current parameters once (these are your baseline; unchanged)
params = struct('pdh',pdh,'pdl',pdl,'BH',BH,'BL',BL,'CH',CH,'CL',CL,...
                'WH',WH,'WL',WL,'cah',cah,'bah',bah,'cal',cal,'bal',bal);

figure('Color','w','Name','Stationary distribution vs z for different \beta (block chain)');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% before the for-ib loop
PIz3d = nan(4, numel(Zvals), numel(betas4));   % <-- add this line

for ib = 1:numel(betas4)
    b = betas4(ib);
    PIz = nan(4, numel(Zvals));                % temp for this beta
    for iz = 1:numel(Zvals)
        z_k = Zvals(iz);
        [~, PI4] = stationary_block_chain(N, z_k, b, params);
        PIz(:, iz) = PI4(:);
    end

    % store the whole slice for this beta
    PIz3d(:, :, ib) = PIz;                     % <-- add this line

    % ... your existing plotting for each panel using PIz ...
end

for ib = 1:numel(betas4)
    b = betas4(ib);
    PIz = nan(4, numel(Zvals));
    for iz = 1:numel(Zvals)
        z_k = Zvals(iz);
        [~, PI4] = stationary_block_chain(N, z_k, b, params); % returns [ (A,H) (A,L) (NA,H) (NA,L) ]
        PIz(:, iz) = PI4(:);
    end

    nexttile; hold on; grid on; box on;
    plot(Zvals, PIz(1,:), '-',  'LineWidth', 2.4, 'Color', colors(1,:)); % (A,H)
    plot(Zvals, PIz(2,:), '-',  'LineWidth', 2.4, 'Color', colors(2,:)); % (A,L)
    plot(Zvals, PIz(3,:), '-',  'LineWidth', 2.4, 'Color', colors(3,:)); % (NA,H)
    plot(Zvals, PIz(4,:), '-',  'LineWidth', 2.4, 'Color', colors(4,:)); % (NA,L)

    yyaxis right
% plot(Zvals, piH_total, 'k--', 'LineWidth', 3);
ylabel('\pi_H (H-defender frequency)'); ylim([0 1]);
yyaxis left


    % --- add this just after the 4 state curves in the nexttile ---
    piH_total = PIz(1,:) + PIz(3,:);                 % total H share
    plot(Zvals, piH_total, 'k--', 'LineWidth', 3);   % overlay (black dashed)

    if ib==1
        legend({'\pi_1 (A,H)','\pi_2 (A,L)','\pi_3 (NA,H)','\pi_4 (NA,L)', '\pi_H=\pi_1+\pi_3'}, ...
               'Location','best'); legend boxoff;
    end


    ylim([0 1]); xlim([Zvals(1) Zvals(end)]);
    xlabel('z (committed H defenders)'); ylabel('Stationary probability');
    title(sprintf('\\beta = %.3g', b));
    if ib==1, legend(stateNames, 'Location','best'); legend boxoff; end
    set(gca,'FontSize',16);
end
% before the for-ib loop
PIz3d = nan(4, numel(Zvals), numel(betas4));   % <-- add this line

for ib = 1:numel(betas4)
    b = betas4(ib);
    PIz = nan(4, numel(Zvals));                % temp for this beta
    for iz = 1:numel(Zvals)
        z_k = Zvals(iz);
        [~, PI4] = stationary_block_chain(N, z_k, b, params);
        PIz(:, iz) = PI4(:);
    end

    % store the whole slice for this beta
    PIz3d(:, :, ib) = PIz;                     % <-- add this line

    % ... your existing plotting for each panel using PIz ...
end

% Optional table at β≈0.1
[~, idxb] = min(abs(betas4-0.1));
T = table(Zvals(:), squeeze(PIz3d(1,:,idxb)+PIz3d(3,:,idxb)).', ...
          'VariableNames', {'z','pi_H_total'});
disp('Total H share (π1+π3) at β≈0.1:'); disp(T);

%% 
% Fixation probability of ONE mutant (the "second" argument below) among ordinary defenders,
% when z zealots permanently play H:
%   N0 = N - z ordinary players evolve;
%   T^+_i  = ((N0 - i)/N0) * ((i + z)/N) * p_{L,H}
%   T^-_i  = ( i / N0)     * ((N0 - i)/N) * p_{H,L}
% with Fermi imitation p_{L,H} = 1/(1 + exp(-beta*(piH - piL))).
function rho = rho_fixation_with_zealots1(N, z, beta, piH, piL)
    N0 = N - z; 
    if N0 < 2
        rho = 0; 
        return; 
    end

    % Fermi pairwise comparison (Eq. 14)
    p_LH = 1.0 / (1.0 + exp(-beta * (piH - piL)));  % L imitates H
    p_HL = 1.0 / (1.0 + exp(-beta * (piL - piH)));  % H imitates L

    % Birth–death kernel (Eq. 15)
    Tplus  = @(i) ((N0 - i) / N0) .* ((i + z) / N) .* p_LH;  % i -> i+1
    Tminus = @(i) ( i       / N0) .* ((N0 - i) / N) .* p_HL; % i -> i-1

    % Fixation probability for a single invading type via Eq. (17):
    % rho = 1 / (1 + sum_{k=1}^{N0-1} prod_{j=1}^{k} T^-_j / T^+_j)
    kmax = N0 - 1;
    if kmax < 1
        rho = 0; 
        return; 
    end

    prodcum = 1.0;
    sumterm = 0.0;
    for k = 1:kmax
        num = Tminus(k);
        den = Tplus(k);
        if den <= 0
            rho = 0; 
            return;
        end
        prodcum = prodcum * (num / den);
        sumterm = sumterm + prodcum;
    end
    rho = 1.0 / (1.0 + sumterm);
end

%%  % Coupled attacker/defender Markov chain with z committed H defenders.
function [pi_block, PI4] = stationary_block_chain(N, z, beta, params)

    pdh = params.pdh;  pdl = params.pdl;
    BH  = params.BH;   BL  = params.BL;
    CH  = params.CH;   CL  = params.CL;
    WH  = params.WH;   WL  = params.WL;
    cah = params.cah;  bah = params.bah;
    cal = params.cal;  bal = params.bal;

    if ~isfield(params,'wA'),     params.wA = 0.5;    end
    if ~isfield(params,'betaA'),  params.betaA = beta;end
    wA    = params.wA;     wD = 1 - wA;
    betaA = params.betaA;  betaD = beta;

    % --- per-interaction payoffs (constant across i) ---
    fA_H  = -cah + bah*(1 - pdh);
    fA_L  = -cal + bal*(1 - pdl);
    fH_A  =  pdh*BH - CH - (1 - pdh)*WH;
    fL_A  =  pdl*BL - CL - (1 - pdl)*WL;
    fH_NA =  BH - CH;
    fL_NA =  BL - CL;

    % --- state space (A,i) and (NA,i), i=0..N0 ---
    N0 = N - z; 
    if N0 < 0, error('z cannot exceed N'); end
    nStates = 2*(N0+1);
    idxA  = @(i) (i+1);
    idxNA = @(i) (N0+1) + (i+1);

    I=[]; J=[]; V=[];

    for a = [1 0] % 1=A, 0=NA
        for i = 0:N0
            s = (a==1)*idxA(i) + (a==0)*idxNA(i);

            % composition incl. zealots
            yH = (i + z)/N;  yL = 1 - yH;

            % defender payoffs given attacker state
            if a==1, piH=fH_A;  piL=fL_A;
            else     piH=fH_NA; piL=fL_NA; end

            % defender imitation (Eq. 15)
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

            % attacker imitation (Eq. 16)
            piA  = yH*fA_H + yL*fA_L;   piNA = 0;
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

    % --- aggregate to 4 macro-states ---
    PI4 = zeros(1,4); % [A,H A,L NA,H NA,L]
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

function val = local_SW_from_PI4(N, z, beta, params, Wvec)
    [~, PI4] = stationary_block_chain(N, z, beta, params);  % 1×4
    val = PI4 * Wvec.';                                     % scalar
end

%% Heatmaps: each π_i (A,H / A,L / NA,H / NA,L) vs (z, beta) % 
Zvals_hm  = 0:10;                       % committed H defenders
beta_grid = logspace(-1, 1, 8);        % selection intensities
nZ = numel(Zvals_hm); nB = numel(beta_grid);

params_base = struct('pdh',pdh,'pdl',pdl,'BH',BH,'BL',BL,'CH',CH,'CL',CL, ...
                     'WH',WH,'WL',WL,'cah',cah,'bah',bah,'cal',cal,'bal',bal);

PI_hm = nan(4, nZ, nB);                 % [state, z, beta]
for jb = 1:nB
    b = beta_grid(jb);
    for iz = 1:nZ
        z_k = Zvals_hm(iz);
        [~, PI4] = stationary_block_chain(N, z_k, b, params_base); % [(A,H) (A,L) (NA,H) (NA,L)]
        PI_hm(:, iz, jb) = PI4(:);
    end
end

% ---- 2x2 heatmaps with shared labels & one colorbar (compact, not touching) ----
stateNames = {'(A,H)','(A,L)','(NA,H)','(NA,L)'};
beta_ticks = [0.1 0.2 0.5 1 2 5 10];
yticks_log = log10(beta_ticks);  % because we plot log10(beta)
yticklabs  = string(beta_ticks);

fig = figure('Color','w','Name','Stationary probabilities vs (z, \beta)');

t = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact'); % small gap between subplots

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

% Shared colorbar — attach to last axis, then stretch it to full height
cb = colorbar(axs(end), 'eastoutside');
cb.Label.String  = 'Stationary Distribution';
cb.Label.FontSize = 28;
cb.Label.FontWeight = 'bold';

% Adjust colorbar to span the full layout height
cb.Layout.Tile = 'east';

% ----- total H share heatmap -----
figure('Color','w','Name','Total H-defender frequency \pi_H vs (z, \beta)');
PIH = squeeze(PI_hm(1,:,:)+PI_hm(3,:,:))';  % [beta,z]
imagesc(Zvals_hm, log10(beta_grid), PIH);
axis xy; caxis([0 1]); colormap(parula); colorbar;
xlabel('Committed defenders  z'); ylabel('Selection intensity  \beta');

beta_ticks_l = [0.1 0.2 0.5 1 2 5 10];
beta_labels  = string(beta_ticks_l);
yticks(log10(beta_ticks_l));
yticklabels(beta_labels);

xlabel('Committed defenders  z','FontSize',28,'FontWeight','bold');
ylabel('Selection intensity  \beta','FontSize',28,'FontWeight','bold');
cb2 = colorbar('Location','eastoutside');
cb2.Label.String = 'Stationary Distribution';
cb2.Label.FontSize = 28;
cb2.Label.FontWeight = 'bold';
cb2.Label.Position = [cb2.Label.Position(1)+2, cb2.Label.Position(2), cb2.Label.Position(3)];
set(gca,'YDir','normal','FontSize',20); grid on;
title('Frequency of H defenders');

%% %% %% %% ============================================================
%  Monte Carlo scatter: π_i (x) vs. π_H (y)
% =============================================================
tic
rng(12);
numGames = 10000;
betas_MC = [1e-3, 0.1, 1, 10];
N_MC = 100;
epsIneq = 1e-3;

figure('Color','w','Name','Stationary distribution components vs H-defender frequency');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
stateNames = {'(A,H)','(A,L)','(NA,H)','(NA,L)'};
colors = lines(4);

for ib = 1:numel(betas_MC)
    b = betas_MC(ib);
    PI4_all = nan(numGames,4);
    piH_all = nan(numGames,1);

    for t = 1:numGames
        % --- random parameters with inequalities ---
        pdl = rand; pdh = min(1,max(pdl+epsIneq,rand));
        BH = 0.4 + 0.6*rand; BL = max(0,min(BH-epsIneq,rand));
        CH = 0.3 + 0.7*rand; CL = max(0,min(CH-epsIneq,rand));
        WL = 0.1 + 0.5*rand; WH = max(WL+epsIneq,min(1,WL+0.4*rand));
        bal = 2*rand; bah = max(bal+epsIneq,2*rand);
        cal = rand;   cah = max(cal+epsIneq,rand);
        z = randi([0,20]);

        params = struct('pdh',pdh,'pdl',pdl,'BH',BH,'BL',BL,'CH',CH,'CL',CL,...
                        'WH',WH,'WL',WL,'cah',cah,'bah',bah,'cal',cal,'bal',bal);
        if z>=N_MC, z=N_MC-1; end

        try
            [~, PI4] = stationary_block_chain(N_MC, z, b, params);
            PI4_all(t,:) = PI4;
            piH_all(t) = PI4(1) + PI4(3);    % total H defenders
        catch
            continue
        end
    end

    % --- plot scatter of each π_i vs π_H ---
    nexttile; hold on; grid on; box on;
    for i = 1:4
        scatter(PI4_all(:,i), piH_all, 8, 'filled', 'MarkerFaceColor', colors(i,:));
    end
    xlabel('Stationary distribution \pi_i');
    ylabel('\pi_H (H-defender frequency)');
    title(sprintf('\\beta = %.3g', b));
    legend(stateNames, 'Location','bestoutside'); legend boxoff;
    xlim([0 1]); ylim([0 1]); set(gca,'FontSize',18);
end
toc

%% %% === Welfare vs committed defenders (SW_D, SW_A, SW) ===
% Uses current N, beta, params, and the per-state payoffs:
% fH_A, fL_A, fH_NA, fL_NA, fA_H, fA_L, fNA_H, fNA_L

Z = 0:min(10, N-1);                   % sweep z from 0 up to 10

% Per-state welfare vectors (order: [(A,H) (A,L) (NA,H) (NA,L)])
W_def = [fH_A,  fL_A,  fH_NA,  fL_NA];        % defender welfare per state
W_att = [fA_H,  fA_L,  fNA_H,  fNA_L];        % attacker welfare per state
W_tot = W_def + W_att;                         % total welfare per state

% Storage
SWD = zeros(size(Z));    % defender welfare
SWA = zeros(size(Z));    % attacker welfare
SWT = zeros(size(Z));    % total welfare
piH = zeros(size(Z));    % total H share (for reference)

for k = 1:numel(Z)
    zk = Z(k);
    [~, PI4] = stationary_block_chain(N, zk, beta, params);  % 1x4 stationary dist.
    SWD(k)   = PI4 * W_def.';     % scalar
    SWA(k)   = PI4 * W_att.';     % scalar
    SWT(k)   = PI4 * W_tot.';     % scalar
    piH(k)   = PI4(1) + PI4(3);   % (A,H)+(NA,H)
end

% ---- Plot: all three welfare curves ----
figure('Color','w','Name','Welfare vs committed defenders z');
hold on; grid on; box on;
p1 = plot(Z, SWD, 'b-o','LineWidth',4,'MarkerSize',5,'DisplayName','SW_D(z)  (defenders)');
p2 = plot(Z, SWA, 'r-s','LineWidth',4,'MarkerSize',5,'DisplayName','SW_A(z)  (attackers)');
p3 = plot(Z, SWT, 'k-^','LineWidth',4,'MarkerSize',5,'DisplayName','SW(z) = SW_D + SW_A');

% Optional: show total H share on right axis (remove this block if not needed)
yyaxis right
% p4 = plot(Z, piH, 'g-','LineWidth',4,'DisplayName','\pi_H');
ylim([0 1]); ylabel('\pi_H (H-defender fraction)');
yyaxis left

xlabel('Committed defenders  z');
ylabel('Welfare');
title('Defender, Attacker, and Total Welfare vs committed defenders z');
legend([p1 p2 p3], 'Location','best', 'FontSize',14);
set(gca,'FontSize',28,'FontWeight','bold');

% ---- Display computed welfare values ----
fprintf('\n=== Welfare summary (beta = %.3f, N = %d) ===\n', beta, N);
fprintf('   z\tSW_D(z)\t\tSW_A(z)\t\tSW_T(z)\n');
fprintf('---------------------------------------------\n');
for k = 1:numel(Z)
    fprintf('%4d\t% .6f\t% .6f\t% .6f\n', Z(k), SWD(k), SWA(k), SWT(k));
end
fprintf('---------------------------------------------\n');
fprintf('Max SW_D = %.6f at z = %d\n', max(SWD), Z(SWD == max(SWD)));
fprintf('Max SW_T = %.6f at z = %d\n\n', max(SWT), Z(SWT == max(SWT)));

%% --- Random games (10,000), β = 1, histograms of attack/defence frequencies
tic
rng(12);
numGames = 10000;
N_MC = 100;
beta  = 1;
epsIneq = 1e-3;

% Storage
pi_A_total  = nan(numGames,1);   % attackers: (A,H)+(A,L)
pi_NA_total = nan(numGames,1);   % non-attackers: (NA,H)+(NA,L)
pi_H_total  = nan(numGames,1);   % H-defenders: (A,H)+(NA,H)
pi_L_total  = nan(numGames,1);   % L-defenders: (A,L)+(NA,L)

for t = 1:numGames
    % --- random parameters with inequalities ---
    pdl = rand;                         pdh = min(1, max(pdl + epsIneq, rand));
    BH  = 0.4 + 0.6*rand;               BL  = max(0, min(BH - epsIneq, rand));
    CH  = 0.3 + 0.7*rand;               CL  = max(0, min(CH - epsIneq, rand));
    WL  = 0.1 + 0.5*rand;               WH  = max(WL + epsIneq, min(1, WL + 0.4*rand));
    bal = 2*rand;                       bah = max(bal + epsIneq, 2*rand);
    cal = rand;                         cah = max(cal + epsIneq, rand);
    z   = randi([0,20]);                if z >= N_MC, z = N_MC - 1; end

    params = struct('pdh',pdh,'pdl',pdl,'BH',BH,'BL',BL,'CH',CH,'CL',CL, ...
                    'WH',WH,'WL',WL,'cah',cah,'bah',bah,'cal',cal,'bal',bal);

    try
        [~, PI4] = stationary_block_chain(N_MC, z, beta, params);  % [(A,H) (A,L) (NA,H) (NA,L)]
        pi_A_total(t)  = PI4(1) + PI4(2);
        pi_NA_total(t) = PI4(3) + PI4(4);
        pi_H_total(t)  = PI4(1) + PI4(3);
        pi_L_total(t)  = PI4(2) + PI4(4);
    catch
        continue
    end
end

% Drop failed runs
ok = ~isnan(pi_A_total);
pi_A_total  = pi_A_total(ok);
pi_NA_total = pi_NA_total(ok);
pi_H_total  = pi_H_total(ok);
pi_L_total  = pi_L_total(ok);
% ---- Histogram 1: Attack vs Non-attack frequencies ----
figure('Color','w','Name','Attack vs Non-attack frequency (β = 1)');
edges = 0:0.02:1;
hold on; grid on; box on;

h1 = histogram(pi_A_total,  edges, ...
    'FaceColor',[0.85 0.2 0.2], 'EdgeColor','k', 'FaceAlpha',0.7);
h2 = histogram(pi_NA_total, edges, ...
    'FaceColor',[0.2 0.2 0.85], 'EdgeColor','k', 'FaceAlpha',0.5);

xlim([0 1]); ylim([0 2000]);
xlabel('Frequency of Attack / Non-Attack', 'FontSize',20, 'FontWeight','bold');
ylabel('Number of Games', 'FontSize',20, 'FontWeight','bold');
title('Distribution of Stationary Attack vs Non-Attack Frequencies', ...
      'FontSize',20, 'FontWeight','bold');
legend({'Attackers  (\pi_{A,tot})', 'Non-Attackers  (\pi_{NA,tot})'}, ...
       'Location','northoutside', 'Orientation','horizontal', ...
       'FontSize',18, 'FontWeight','bold');
set(gca,'FontSize',24,'FontWeight','bold','LineWidth',1.5); 
% ---- Histogram 2: H vs L defence frequencies ----
figure('Color','w','Name','H vs L defence frequency (β = 1)');
hold on; grid on; box on;

h3 = histogram(pi_H_total, edges, ...
    'FaceColor',[0.15 0.6 0.15], 'EdgeColor','k', 'FaceAlpha',0.7);
h4 = histogram(pi_L_total, edges, ...
    'FaceColor',[0.85 0.65 0.15], 'EdgeColor','k', 'FaceAlpha',0.5);

xlim([0 1]); ylim([0 7000]);
xlabel('Frequency of Defence (H / L)', 'FontSize',20, 'FontWeight','bold');
ylabel('Number of Games', 'FontSize',20, 'FontWeight','bold');
title('Distribution of Stationary H vs L Defence Frequencies', ...
      'FontSize',20, 'FontWeight','bold');
legend({'High-Defence  (\pi_{H,tot})', 'Low-Defence  (\pi_{L,tot})'}, ...
       'Location','northoutside', 'Orientation','horizontal', ...
       'FontSize',18, 'FontWeight','bold');
set(gca,'FontSize',24,'FontWeight','bold','LineWidth',1.5);

toc


      