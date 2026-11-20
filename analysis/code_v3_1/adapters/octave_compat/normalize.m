function B = normalize(A, method, p)
% MATLAB-compatible normalize(A,'norm',p) for core Octave, which does not ship
% this Statistics-toolbox function.  Only the one call the authors' TMI code
% makes is supported; anything else errors instead of silently differing.
%
% MATLAB: normalize(A,'norm',p) scales along the first non-singleton dimension
% (columns of a matrix) so each vector has unit p-norm.
  if nargin < 3 || ~ischar(method) || ~strcmpi(method, 'norm')
    error('normalize (compat shim): only normalize(A,''norm'',p) is supported');
  end
  dim = 1;
  if size(A, 1) == 1
    dim = 2;
  end
  if p == 1
    scale = sum(abs(A), dim);
  elseif isinf(p)
    scale = max(abs(A), [], dim);
  else
    scale = sum(abs(A) .^ p, dim) .^ (1 / p);
  end
  B = bsxfun(@rdivide, A, scale);
end
