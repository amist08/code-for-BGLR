function [W,index,newfea,lossFun]=BGLR(X,k,m,h,beta,eta,maxIter)
   %% input:
       %    data matrix:X,dxn
       %    feature number:k
       %    projection dimension:m
       %    the number of anchors:h
       %    regularization parameter:beta,eta
    %% output:
       %    newfea:a feature subset containing k features
    %% Initialization of relevant matrix
    disp('Initialization of relevant matrix')
    [d,n]=size(X);
    W=rand(d,m);
    % Normalize W to satisfy WTW = I
%     W = W / sqrtm(W' * W); % Normalize W using square root of its covariance matrix
    P=zeros(n,h);
    lossFun=zeros(1,maxIter);
    E1=zeros(n,n);
    E2=zeros(h,h);
    I=eye(d);
    I1=ones(d);
   di=zeros(1,n);
   %calculate the similarity between features
    featuresX=normalize(X'); %归一化函数：将数据归一化，使其均值为 0，标准差为 1，如果 A 是矩阵、表或时间表，则 normalize 分别对数据的每个列进行运算
    S1=featuresX'*featuresX;
    S1=S1-diag(diag(S1));
    %% Anchor Selection and Bipartite graph Construction
    disp('Anchor Selection and Bipartite graph Construction')
    cls_num=h;
   [Z]=get_Bipartite(X,cls_num,1);%初始化二部图、锚点
   %% 迭代更新相关变量
   for i=1:maxIter
      fprintf('第%d次迭代\n',i);
      disp('update P')
       %% update P
       parfor r=1:n
           for l=1:m
               di(r)=di(r)+exp(-(norm(W'*X(:,r)-W'*Z(:,l),2)^2)/eta);
           end
           for j=1:h
               P(r,j)=(exp(-(norm(W'*X(:,r)-W'*Z(:,j),2)^2)/eta))/(di(r)+eps);
           end
       end
       %% update W
       disp('update W')
       new_X=[X Z];   
       S=[E1 P;P' E2];
       [L_S]=Lap(S);
       A_X=new_X*L_S*new_X'+beta*S1*I1;
       A_X(isnan(A_X) | isinf(A_X)) = 1;
       [~,u]=eig(A_X);
       lambda=real(max(diag(u)));
       G=lambda*I-A_X;
       if(rank(G)<=m)
         Q1=zeros(d,k);
         [~,idx1] = sort(diag(G), 'descend');   % 对G进行降序排序，并返回排序后的结果以及对应的索引
         for b=1:d
             for s=1:k
                 if(b==idx1(s))
                     Q1(b,s)=1;
                 end
             end
         end
         M1=Q1'*G*Q1;
         [U1, ~] = eigs(M1, m,'lm');
         W=Q1*U1;
       end
       if(rank(G)>m) %repeat util convenge
         loss=zeros(1,20);
         for time=1:20
             E=G*W*pinv(W'*G*W)*W'*G;
             Q2=zeros(d,k);
             [~,idx2] = sort(diag(E), 'descend');   % 对P进行降序排序，并返回排序后的结果以及对应的索引
             for b=1:d
                 for s=1:k
                     if(b==idx2(s))
                         Q2(b,s)=1;
                     end
                 end
             end
%              M2=G*W;
%              U2=M2*pinv(M2'*M2)*M2';
             M2=Q2'*G*Q2;
             [U2, ~] = eigs(M2, m,'lm');
             W=real(Q2*U2);
             loss(time)=trace(W'*E*W);
         end
       end
       %% 计算损失函数
       disp('calculate loss')
       tem1=beta*trace(W'*S1*I1*W);
       tem2=trace(W'*new_X*L_S*new_X'*W);
       tem3=eta*P.*log(P);
       lossFun(i)=tem1+tem2+sum(tem3,'all');
       if(i>10)
           if(abs((lossFun(i)-lossFun(i-1)))/lossFun(i)<0.00001)
               break;
           end
       end  
   end
   [index,~]=find(W);
   newfea=X(index(1:k),:);
end