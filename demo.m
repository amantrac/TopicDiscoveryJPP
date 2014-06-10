%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright (c) 2014 Yahoo! Inc.
%Copyrights licensed under the MIT License. See the accompanying LICENSE file for terms.
%Author: Amin Mantrach  - amantrac at yahoo - inc dot com - http://iridia.ulb.ac.be/~amantrac/
%This is demo file on how to use the JPP decomposition,
%it will produce the final scores in terms of micro F1, macro F1, NAP and NDCG
%the data set used is the TDT2 data set publicaly available from: http://www.nist.gov/speech/tests/tdt/tdt98/index.htm, 
%We are using the matlab version available here: http://www.cad.zju.edu.cn/home/dengcai/Data/TextData.html
%The demo is configured to use 6 topics (k=6, you can change it)
%it initialize the system the first week, using NMF
%then it computes the result the remaining week from 2 to 26.
%intermediary results are displayed at each step for
%JPP, using NMF on the current timestamp (tmodel in the paper), and NMF
%on a fixed starting period timestamp (fix model).
%The demo file use lambda 10000000 this can be changed.
%In case of news, we observed that for a periof of one day, putting high
%value of lambda is the best, as we put emphasis on the past
%If you have a prior on high periodicity of the events, use value =1
%if you don't know, you can do a simple cross-val experimentation 
%a set lambda using a validation set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
%We load the data such that we have 3 matrices
%X doc x words, T doc x time step and Y doc x label


load TDT2.mat;
X=fea;
load T.mat;
Y=[];
k=6; %We fix the nb of top classes to track
for v =[1:k]
    Y  = [Y gnd==v];
end
X = X(find(sum(Y,2)),:);
Y = Y(find(sum(Y,2)),:);
T = T(find(sum(Y,2)),:);

%load ynewsData;
%[s i ]= sort(sum(Y),'descend');
%k=30;
%Y = Y(:,i(1:k));
%X = X(find(sum(Y,2)),:);
%Y = Y(find(sum(Y,2)),:);
%T = T(find(sum(Y,2)),:);



 numlambda=0;


%flag variable
JPPflag=true;
MR = [];
MRO = [];
MRbaseline =[];
MRfix =[];



regl1nmf = 0.0005;

regl1jpp = 0.05;

epsilon = 0.01;

maxiter = 100;
for lambda = [10000000]

numlambda = numlambda+1;


%the start time period used for init of W(1) and H(1), using normal NMF
for start= [1]


t = find(sum(T(:,start),2));
Xt = X(t,:);
idf = log(size(Xt,1)./(sum(Xt>0)+eps));
IDF = spdiags(idf',0,size(idf,2),size(idf,2));
Xtfidf = L2_norm_row(Xt*IDF);

%call NMF with L1 norm
[W H] = NMF(Xtfidf, k, regl1nmf, epsilon, maxiter, false);
Hfixmodel = L2_norm_row(H);
Hbaseline2= H;
Honline=H;
HA=H;

%number of period we consider
finT = size(T,2);


%for all the consecutive periods
for weeks = [start+1:finT]

fprintf('\n=========================\n');
fprintf('week number %i:\n',weeks);
fprintf('----------------\n');
%compute the grountruth as the top 10 words of the center of mass each label set    

nbtopicalwords=10;
t = find(sum(T(:,start:weeks),2));
Xt = X(t,:);
idf = log(size(Xt,1)./(sum(Xt>0)+eps));
IDF = spdiags(idf',0,size(idf,2),size(idf,2));
Xtfidf = L2_norm_row(Xt*IDF);
Yt = Y(t,:);
Htrue = Yt'*Xtfidf;
Htrue = L2_norm_row(Htrue);
[void I]=sort(Htrue,2,'descend');

for i=1:size(Htrue,1)
      Htrue(i,I(i,1:nbtopicalwords))=1;
      Htrue(i,I(i,nbtopicalwords+1:end))=0;
end




    
    
    t = find(sum(T(:,[weeks]),2));
    Xt = X(t,:);
    idf = log(size(Xt,1)./(sum(Xt>0)+eps));
    IDF = spdiags(idf',0,size(idf,2),size(idf,2));
    Xtfidf = L2_norm_row(Xt*IDF);
    if(size(Xtfidf,1)==0)
        continue;
    end
     
    
 
    
    Ho=H;
    
    if(JPPflag),
      fprintf('computing JPP decomposition...');
      [W, H, M, ~] = JPP(Xtfidf, Ho, size(Ho,1), lambda, regl1jpp,  epsilon, maxiter, false);
   
    end
 
    if(numlambda==1)'
        fprintf('[ok]\ncomputing NMF decomposition...'); 
        [void Hbaseline2] = NMF(Xtfidf, k,regl1nmf, epsilon, maxiter, false);
        fprintf('[ok]\n');
        Hbaseline = L2_norm_row(Hbaseline2);          
    end
   
   
    
	Hev = L2_norm_row(H);
    if(JPPflag),
        Hmax = [];
        for i=[1:size(Htrue,1)]
         max = Htrue(i,:)*Hev(1,:)';
         maxi = 1;
         for j=[2:size(Hev,1)]
            val =  Htrue(i,:)*Hev(j,:)';
            if (max < val)
                max = val;
                maxi = j;
            end
         end
         Hmax = [Hmax; Hev(maxi,:)];
        end
    end
    
    if(numlambda==1),
                Hmaxbaseline = [];
                for i=[1:size(Htrue,1)]
                 max = Htrue(i,:)*Hbaseline(1,:)';
                 maxi = 1;
                 for j=[2:size(Hbaseline,1)]
                    val =  Htrue(i,:)*Hbaseline(j,:)';
                    if (max < val)
                        max = val;
                        maxi = j;
                    end
                 end
                 Hmaxbaseline = [Hmaxbaseline; Hbaseline(maxi,:)];
                end


                Hmaxfix = [];
                for i=[1:size(Htrue,1)]
                 max = Htrue(i,:)*Hfixmodel(1,:)';
                 maxi = 1;
                 for j=[2:size(Hfixmodel,1)]
                    val =  Htrue(i,:)*Hfixmodel(j,:)';
                    if (max < val)
                        max = val;
                        maxi = j;
                    end
                 end
                 Hmaxfix = [Hmaxfix; Hfixmodel(maxi,:)];
                end
                                
    end
     R=[];
    
     
    
    if(JPPflag),
         [NDCG] = performanceNDCG(Hmax,Htrue);
        MR = [MR; [NDCG]];
        fprintf('JPP  scores - NDCG: %f\n',NDCG);

    end
    
    
    
    if (numlambda==1),
  	  Rbaseline = [];
  	  [NDCG] = performanceNDCG(Hmaxbaseline,Htrue);
 	   MRbaseline = [MRbaseline;[NDCG] ];
        fprintf('t-model  scores -  NDCG: %f\n',NDCG);

    
  	  Rfix = [];
  	  [NDCG] = performanceNDCG(Hmaxfix,Htrue);
  	  MRfix = [MRfix;  [NDCG]];
      fprintf('fix-model  scores - NDCG: %f\n',NDCG);
    end
    fprintf('=========================\n');

end %end for weeks


end %for start 



end %for lambda
mmr = mean(MR);
fprintf('JPP Avg scores NDCG: %f\n',mmr(1));
mmr = mean(MRbaseline);
fprintf('t-model Avg NMF scores NDCG: %f\n',mmr(1));
mmr = mean(MRfix);
fprintf('fix-model Avg NMF scores NDCG: %f\n',mmr(1));










