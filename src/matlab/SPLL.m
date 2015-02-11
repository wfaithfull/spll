function [Change,pst,st] = SPLL(W1,W2,PARAM)
%
% Detects a change in a multivariate data stream using the log-likelihood
% method
%
% [ Kuncheva L. I., Change detection in streaming multivariate data using
% likelihood detectors, IEEE Transactions on Knowledge and Data
% Engineering, (in press)
% http://pages.bangor.ac.uk/~mas00a/papers/lktkde11a.pdf ]
%
% Input:
%     W1 = Data in Window 1:
%          matrix of size M1-by-n (M1 time points, n features)
%     W2 = Data in Window 2:
%          matrix of size M2-by-n (M2 time points, n features)
%     PARAM(1) = K - the number of clusters, default K = 3
%
% Output
%      Change = indicator: 0 - no change, 1 - change
%      st = change statistic
%%
%========================================================================
% (c) Fox's Classification Toolbox                                  ^--^
% v.1.0 2012 -----------------------------------------------------  \oo/
%                                                                    \/
% Clustering within a window of size M1/M2 into K clusters
 
 
if nargin == 2 || isempty(PARAM)
    K = 3; % Predefined number of clusters
else
    K = PARAM;
end
 
[Ch(1),ps(1),s(1)] = log_LL(W1,W2,K);
[Ch(2),ps(2),s(2)] = log_LL(W2,W1,K);
 
[~,ind] = max(s);
Change = Ch(ind);st = s(ind); pst = ps(ind);
 
end
 
function [Change,pst,st] = log_LL(W1,W2,K)
 
 
[M1,n1] = size(W1);
[M2,n2] = size(W2);
 
if n1 == n2
    n = n1;
else
    error('The number of features must be the same for both windows')
end

% in order to be predictable on identical data, start with identical points
[labels,means] = kmeans(W1,K,'Start',W1(1:K,:),'EmptyAction','singleton');
% uses stats toolbox
% Calculate the REFERENCE distribution from Window 1
 
SC = [];
for k = 1:K
    ClassIndex = find(labels == k);
    ClassPriors(k) = numel(ClassIndex);
    if numel(ClassIndex) <= 1 % singleton cluster
        sc = zeros(n);
    else
        %sc = cov(W1(labels == k,:));
        sc = diag(var(W1(labels==k,:)));
    end
    SC = [SC;sc(:)'];
end
ClassCount = ClassPriors;
ClassPriors = ClassPriors/M1;
scov = sum(SC.*repmat(ClassPriors',1,n^2),1);
 
% Assuming the same full covariance
% Pooled (weighted by the priors)
 
scov = reshape(scov,n,n);
z = diag(scov);

 
indexvarzero = z < eps;
if isempty(indexvarzero)
    invscov = inv(scov);
else
    z(indexvarzero) = min(z);
    invscov = diag(1./z);
end

% Calculate the MATCH from Window 2
% Distance between cluster means
for j = 1:M2
    xx = W2(j,:);
    
    for ii = 1:K
        if ClassCount(ii) > 0
            DistanceToMeans(ii) = (means(ii,:) - xx)...
                * invscov * (means(ii,:) - xx)';
        else
            DistanceToMeans(ii) = inf;
        end
    end
    
    LogLikelihoodTerm(j) = min(DistanceToMeans);
    
    % scale
end

st = mean(LogLikelihoodTerm);
pst = min(chi2cdf(st,n), 1-chi2cdf(st,n));
Change = pst < 0.05;
 
end


