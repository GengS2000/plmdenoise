function [Q] = embed_q(z, theta, S)
% Verbatim copy of the nested function q() inside the authors'
% EMBED/Scripts/source_codes/TMI_masked_samples.m.  It is duplicated here only
% because a MATLAB nested function cannot be called from outside its parent;
% the body below is the authors' code with `N` renamed to `S` and `o` derived
% from the inputs instead of the enclosing scope.  Nothing else is changed:
%
%     function [Q] = q(z,theta)
%         Q = exp(-theta*z);
%         for n=1:N
%             Q([(n-1)*o+1:o*n],:) = normalize(Q([(n-1)*o+1:o*n],:),'norm',1);
%         end
%         Q = Q/N;
%     end
%
% Returns the Gibbs-Boltzmann probabilities: per (subject, time) compositions
% divided by the number of subjects, exactly as the fit uses them.
  O = size(theta, 1);
  o = O / S;
  Q = exp(-theta * z);
  for n = 1:S
    Q([(n - 1) * o + 1 : o * n], :) = normalize(Q([(n - 1) * o + 1 : o * n], :), 'norm', 1);
  end
  Q = Q / S;
end
