function y = nansum(x, dim)
% MATLAB-compatible nansum for core Octave: sum ignoring NaN.
  x(isnan(x)) = 0;
  if nargin < 2
    y = sum(x);
  else
    y = sum(x, dim);
  end
end
