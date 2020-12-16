% ��������Ϊ label����
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
        Xi = X(:,idxi);     % �ҵ�����ÿһ�������
        ceni = mean(Xi,2);  % ��������
        center(:,ii) = ceni;% ������
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; % ÿ�����������������ĵľ���
        sumd(ii,1) = sum(d2c);  % ÿ�����������������ĵľ��� ��
end
 f(1)= sum(sumd);  % ��ʼ����ȥ��Ŀ�꺯��
 
 
 
    
while any(label ~= last)  % ���������ֹͣ��     
    last = label;
    
    BB = X*F;
    FF = F'*F ;                   % �Խ�����Ԫ��Ϊ����ÿ�����������
    aa=diag(FF);
    [~,M]=max(F,[],2);
    
 for i = 1:n    % ����ÿ�ζ�  
  
    m = M(i) ;    % �ҵ���i��Ϊ1Ԫ�ص�λ��      
    if aa(m)==1
        continue;
    end 
    for k = 1:c 
        b = BB(:,k);       % BB(:,k) = X*F(:,k);   �����ڵ�K�������zhi�ͣ�
        a = aa(k);%a=FF(k,k);       % a= F(:,k)'*F(:,k)Ϊ FF = F'*F��k���Խ�Ԫ��(���ڵ�k�����������) ���ֻ��һ�࣬a=1, Ҳ�����NaN 
        if k == m            % ������㣬�϶���һ�в���   
           delta(k) = b'*b / a - ( b'*b- 2 * X(:,i)'* b + X(:,i)'* X(:,i)) / (a -1); % ֻ���ܳ���a-1=0
        else  
           delta(k) = ( b'*b + 2 * X(:,i)'* b + X(:,i)'* X(:,i)) / (a +1) -  b'*b / a; % ������ֿ��࣬a=b=0. 0/0 NAN  
        end         
    end  
    [~,p] = max(delta);      % ���F�ĵ�I�е�����ֵ �ҳ�ʹ��Ŀ�꺯����������(����������NAN  ֱ�Ӻ�����һ��)
    if p~=m
         F(i,m)=0;
         F(i,p)=1;       % ����F      
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
        Xi = X(:,idxi);      % �ҵ�����ÿһ�������
        ceni = mean(Xi,2);   % ��������
        center1(:,ii) = ceni;% ������
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; % ÿ�����������������ĵľ���
        sumd(ii,1) = sum(d2c);  % ÿ�����������������ĵľ��� ��
    end
    f(iter_num+1) = sum(sumd) ;     % 

    
end

 
    
 O=min(f);    
[~,Y]=max(F,[],2);
end
