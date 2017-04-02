X = load('statlog.heart.data');
y = X(:,end);
y(y==2) = -1;
X = X(:,1:end-1);
n = size(X,1);

% Add bias and standardize
X = [ones(n,1) standardizeCols(X)];
d = size(X,2);

% Set regularization parameter
lambda = 1;

% Initialize dual variables
z = zeros(n,1);

% Some values used by the dual
YX = diag(y)*X;
G = YX*YX';

maxIter = 2000; % a large number to get precise results
for t = 1:n*maxIter
    i = randi(n);
    z(i) = 0;
    Dg(i) = 1 - G(i,:)*z/lambda; 
    z(i) = Dg(i)*lambda/G(i,i);
    
    if z(i) < 0 
        z(i) = 0;
    elseif z(i) > 1 
        z(i) = 1;
    end
end

% Convert from dual to primal variables
w = (1/lambda)*(YX'*z);
% Evaluate dual objective:
D = sum(z) - (z'*G*z)/(2*lambda);
% Evaluate primal objective:
P = sum(max(1-y.*(X*w),0)) + (lambda/2)*(w'*w);
fprintf('D is %f, P is %f. There are %f support vectors\n',D,P,sum(z~=0))