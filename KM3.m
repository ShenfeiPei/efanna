% 收敛条件为 label不变
function [F, Y, O, iter_num, f] = KM(X, label,c,maxiter)
% X d*n data
% c is the number of clusters
% F is initial indicator matrix n*c

[d,n] = size(X);
F = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix 
last = 0;
iter_num = 0;


for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);     % 找到属于每一类的样本
        ceni = mean(Xi,2);  % 求类中心
        center(:,ii) = ceni;% 类中心
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; % 每个类中样本与类中心的距离
        sumd(ii,1) = sum(d2c);  % 每个类中样本与类中心的距离 和
end
 f(1)= sum(sumd);  % 初始化进去的目标函数
 
 
 
    
while any(label ~= last)  % 变量不变就停止了     
    last = label;
    
    BB = X*F;
    FF = F'*F ;                   % 对角线上元素为属于每类的样本个数
    aa=diag(FF);
    [~,M]=max(F,[],2);
    
 for i = 1:n    % 空类每次都  
  
    m = M(i) ;    % 找到第i行为1元素的位置      
    if aa(m)==1
        continue;
    end 
    for k = 1:c 
        b = BB(:,k);       % BB(:,k) = X*F(:,k);   （属于第K类的样本zhi和）
        a = aa(k);%a=FF(k,k);       % a= F(:,k)'*F(:,k)为 FF = F'*F第k个对角元素(属于第k类的样本个数) 如果只有一类，a=1, 也会出现NaN 
        if k == m            % 如果满足，肯定那一列不空   
           delta(k) = b'*b / a - ( b'*b- 2 * X(:,i)'* b + X(:,i)'* X(:,i)) / (a -1); % 只可能出现a-1=0
        else  
           delta(k) = ( b'*b + 2 * X(:,i)'* b + X(:,i)'* X(:,i)) / (a +1) -  b'*b / a; % 如果出现空类，a=b=0. 0/0 NAN  
        end         
    end  
    [~,p] = max(delta);      % 算出F的第I行的最优值 找出使得目标函数增加最多的(如果结果中有NAN  直接忽略那一列)
    if p~=m
         F(i,m)=0;
         F(i,p)=1;       % 更新F      
         BB(:,p)=BB(:,p)+X(:,i); % BB(:,p)=X*F(:,p);
         BB(:,m)=BB(:,m)-X(:,i); % BB(:,m)=X*F(:,m);
         aa(p)= aa(p) +1; %  FF(p,p)=F(:,p)'*F(:,p);
         aa(m)= aa(m) -1; %  FF(m,m)=F(:,m)'*F(:,m)
    end
 end           


%%
[~,label]=max(F,[],2);
   iter_num = iter_num+1;
   [~,Y12]=max(F,[],2);
   for ii=1:c
        idxi = find(Y12==ii);
        Xi = X(:,idxi);      % 找到属于每一类的样本
        ceni = mean(Xi,2);   % 求类中心
        center1(:,ii) = ceni;% 类中心
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; % 每个类中样本与类中心的距离
        sumd(ii,1) = sum(d2c);  % 每个类中样本与类中心的距离 和
    end
    f(iter_num+1) = sum(sumd) ;     % 

    
end

 
    
 O=min(f);    
[~,Y]=max(F,[],2);
end
