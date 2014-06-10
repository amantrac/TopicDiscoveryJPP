%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright (c) 2014 Yahoo! Inc.
%Copyrights licensed under the MIT License. See the accompanying LICENSE file for terms.
%Author : Amin Mantrach - amantrac at yahoo - inc dot com - http://iridia.ulb.ac.be/~amantrac/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [NDCG]=performanceNDCG(P,Y)
% P is the predicted scores (n x k) where k is the number of categories
% Y is the label matrix (n x k) with binary values
[n,k]=size(P);
nL=sum(Y,2);
[void,idx]=sort(P,2,'descend');
NDCG=0;
for i=1:n
     NDCG = NDCG +sum(Y(i,idx(i,:)).*[1 1./log2(2:k)])/sum([1 1./log2(2:nL(i))])/n;
end