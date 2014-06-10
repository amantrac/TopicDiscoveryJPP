%Copyright (c) 2014 Yahoo! Inc.
%Copyrights licensed under the MIT License. See the accompanying LICENSE file for terms.
%Author : Amin Mantrach - amantrac at yahoo - inc dot com - http://iridia.ulb.ac.be/~amantrac/
function [W, H] = NMF(X,  k, lambda, epsilon, maxiter, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% alpha: hyper-paramter of the Laplacian regularization
% lambda: hyper-paramter of the L2 regularization
% 
%
% Optimizes the formulation:
% ||X - W*H||^2  + lambda*[l1 Regularization of W and H]
%
% with multiplicative rules.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fix seed for reproducable experiments
rand('seed', 14111981);

% initilasation
n = size(X, 1);
v1 = size(X, 2);

W  = abs(rand(n, k));
H = abs(rand(k, v1));
I =speye(k,k);
Ilambda = I*lambda;

% constants
trXX = tr(X, X);
trII = tr(Ilambda,Ilambda);

% values for the 1st iteration
WtW = W' * W;
WtX = W' * X;
WtWH = WtW * H;

% iteration counters
itNum = 1;
Obj = 10000000;

prevObj = 2*Obj;

%this improves sparsity, not mandatory.

while((abs(prevObj-Obj) > epsilon) && (itNum <= maxiter)),
     W =  W .* (X*H'./max(W*(H*H')+lambda,eps));
     WtW =W'*W;
     WtX = W'*X;
     H = H .*  (WtX ./ max(WtW*H+lambda,eps));
     prevObj = Obj;
     Obj = computeLoss(X,W,H,lambda, trXX, I,WtW);
     delta = abs(prevObj-Obj);
		if verbose,
            fprintf('It: %d \t Obj: %f \t Delta: %f  \n', itNum, Obj, delta); 
        end	  
 	 itNum = itNum + 1;
end


function [trAB] = tr(A, B)
	trAB = sum(sum(A.*B));
end    

function Obj = computeLoss(X,W,H,reg_norm,trXX, I,WtW)
 %   WtW = W' * W;    
    WH = W * H;    
    tr1 = trXX - 2*tr(X,WH) + tr(WH,WH);    
    tr2 = reg_norm*(sum(sum(H)) + sum(sum(W)));
    Obj = tr1+  tr2;    
end


end
% END
