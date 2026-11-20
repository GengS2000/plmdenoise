% Run the authors' EMBED / TMI on one benchmark dataset (MATLAB or Octave).
%
%   octave --no-gui --quiet embed_runner.m input.csv output.csv diag.json \
%          SOURCE_DIR K learn_z learn_the max_steps min_fitting min_gradients restarts seed
%
% The fit is `TMI_masked_samples` from EMBED/Scripts/source_codes, unchanged.
% The mask marks (subject, visit) samples the fit must ignore -- missing visits
% and, during rank selection, the held-out validation samples -- which is the
% authors' own mechanism for incomplete series.  `embed_q.m` beside this file is
% a verbatim copy of the nested q() inside TMI_masked_samples.m, needed only
% because that nested function cannot be called from outside.
%
% Input CSV rows are subject, visit, time, counts...; a visit that is not part
% of the fit carries -1 in every taxon column.  Output rows carry the same keys
% and the estimated relative composition for every (subject, visit).

% Octave passes script arguments through argv(); MATLAB has no equivalent for
% `-batch`, so both engines also accept a one-argument-per-line file named by
% the EMBED_ARGS_FILE environment variable.
args_file = getenv('EMBED_ARGS_FILE');
if ~isempty(args_file)
  fid = fopen(args_file, 'r');
  args = {};
  line = fgetl(fid);
  while ischar(line)
    args{end + 1} = strtrim(line);
    line = fgetl(fid);
  end
  fclose(fid);
else
  args = argv();
end
if numel(args) < 12
  error('embed_runner: expected 12 arguments, got %d', numel(args));
end
input_path = args{1};
output_path = args{2};
diag_path = args{3};
source_dir = args{4};
K = str2double(args{5});
learn_z = str2double(args{6});
learn_the = str2double(args{7});
max_steps = str2double(args{8});
min_fitting = str2double(args{9});
min_gradients = str2double(args{10});
restarts = str2double(args{11});
seed = str2double(args{12});

here = fileparts(mfilename('fullpath'));
is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if is_octave
  % Compatibility shims for two MATLAB functions the authors' file calls and
  % core Octave does not ship (normalize, nansum). Both reproduce the
  % documented MATLAB behaviour for exactly the call used in TMI.
  addpath(fullfile(here, 'octave_compat'));
  more off;
end
addpath(here);
addpath(source_dir);

raw = dlmread(input_path, ',');
n_taxa = size(raw, 2) - 3;
subjects = unique(raw(:, 1));
visits = unique(raw(:, 2));
S = numel(subjects);
T = numel(visits);
if size(raw, 1) ~= S * T
  error('embed_runner: input is not a complete subject x visit grid');
end

% X is (S*o) x T: subject blocks stacked vertically, time in columns.
X = zeros(S * n_taxa, T);
mask = zeros(S, T);
for row = 1:size(raw, 1)
  s = find(subjects == raw(row, 1));
  t = find(visits == raw(row, 2));
  counts = raw(row, 4:end);
  if all(counts == -1)
    % Placeholder only: mask is zero here, so these columns never enter a
    % gradient. A uniform column keeps the authors' per-column normalisation
    % finite (an all-zero column would divide by zero).
    X((s - 1) * n_taxa + 1 : s * n_taxa, t) = 1;
    mask(s, t) = 0;
  else
    X((s - 1) * n_taxa + 1 : s * n_taxa, t) = counts(:);
    mask(s, t) = 1;
  end
end

% Subject-block normalised data, exactly as TMI normalises it internally; used
% for the masked KL and for the post-hoc convergence statistic below.
Xn = X;
mask_exp = zeros(S * n_taxa, T);
for s = 1:S
  block = (s - 1) * n_taxa + 1 : s * n_taxa;
  Xn(block, :) = normalize(X(block, :), 'norm', 1);
  mask_exp(block, :) = repmat(mask(s, :), n_taxa, 1);
end
Xn = Xn / S;

best_kl = Inf;
best_z = []; best_theta = [];
kls = zeros(1, restarts);
started = tic();
for run = 1:restarts
  if is_octave                       % TMI initialises z and theta with rand()
    rand('state', seed + run);
    randn('state', seed + run);
  else
    rng(seed + run);
  end
  [z, theta, ~] = TMI_masked_samples(X, S, K, mask, learn_z, learn_the, ...
                                     max_steps, min_fitting, min_gradients);
  Q = embed_q(z, theta, S);
  % KL over the samples the fit actually used (the authors' KL includes the
  % masked placeholder columns, which would not be comparable across runs).
  kl = 0;
  for s = 1:S
    block = (s - 1) * n_taxa + 1 : s * n_taxa;
    for t = 1:T
      if mask(s, t) == 1
        p = Xn(block, t); q = Q(block, t);
        keep = p > 0;
        kl = kl + sum(p(keep) .* log(p(keep) ./ q(keep)));
      end
    end
  end
  kls(run) = kl;
  if kl < best_kl
    best_kl = kl; best_z = z; best_theta = theta;
  end
end
elapsed = toc(started);

% Post-hoc convergence statistic: TMI stops when norm(gr_theta)/norm(theta)
% falls below min_gradients but does not return it, so it is recomputed here
% from the authors' own gradient expression at the returned (z, theta).
Qb = embed_q(best_z, best_theta, S);
tmp = mask_exp .* (Qb - Xn);
rel_theta_final = norm(tmp * best_z') / norm(best_theta);
rel_z_final = norm(best_theta' * tmp) / norm(best_z);
converged = rel_theta_final < min_gradients;

Q = embed_q(best_z, best_theta, S) * S;    % undo the 1/S the authors apply
out = zeros(size(raw, 1), 3 + n_taxa);
for row = 1:size(raw, 1)
  s = find(subjects == raw(row, 1));
  t = find(visits == raw(row, 2));
  out(row, 1:3) = raw(row, 1:3);
  out(row, 4:end) = Q((s - 1) * n_taxa + 1 : s * n_taxa, t)';
end
% dlmwrite keeps enough digits for a composition that must sum to 1 to 1e-4.
dlmwrite(output_path, out, 'delimiter', ',', 'precision', '%.10g');

fid = fopen(diag_path, 'w');
fprintf(fid, '{\n');
if is_octave
  fprintf(fid, '  "engine": "octave %s",\n', version());
else
  fprintf(fid, '  "engine": "matlab %s",\n', version());
end
fprintf(fid, '  "rank": %d,\n', K);
fprintf(fid, '  "learn_z": %.10g,\n', learn_z);
fprintf(fid, '  "learn_theta": %.10g,\n', learn_the);
fprintf(fid, '  "max_steps": %d,\n', max_steps);
fprintf(fid, '  "min_fitting": %.10g,\n', min_fitting);
fprintf(fid, '  "min_gradients": %.10g,\n', min_gradients);
fprintf(fid, '  "restarts": %d,\n', restarts);
fprintf(fid, '  "seed": %d,\n', seed);
fprintf(fid, '  "kl_per_restart": [');
fprintf(fid, '%.10g', kls(1));
for run = 2:restarts
  fprintf(fid, ', %.10g', kls(run));
end
fprintf(fid, '],\n');
fprintf(fid, '  "kl_masked_best": %.10g,\n', best_kl);
fprintf(fid, '  "relative_gradient_theta": %.10g,\n', rel_theta_final);
fprintf(fid, '  "relative_gradient_z": %.10g,\n', rel_z_final);
if converged
  fprintf(fid, '  "converged": true,\n');
else
  fprintf(fid, '  "converged": false,\n');
end
fprintf(fid, '  "fitted_samples": %d,\n', sum(mask(:)));
fprintf(fid, '  "input_shape": [%d, %d, %d],\n', S, T, n_taxa);
fprintf(fid, '  "elapsed_seconds": %.6f\n', elapsed);
fprintf(fid, '}\n');
fclose(fid);
