%% preprocessing
load('2013MT60079.mat');
M = double(data_image);
t = data_labels;
[coeff,score,latest,tsquared,explained] = pca(M);
%% to use minimum number of components
sumvar=0;
x=zeros(784);
for i=1:784
    x(i)=i;
end
resvar=zeros(784);
var=zeros(784);
for i=1:784
    for j=1:i
        var(i)=var(i)+explained(j);
    end
    resvar(i)=100-var(i);
end
min=784;
for i=1:784
    if resvar(i)<=10
        min=i;
        break
    end
end
%% apply kmeans in PCA space
N=score(:,1:min);
[idx,c] = kmeans(N,10,'Replicates',1);

%% Loading data and initialising variables

k = 10 ;%number of clusters
n = 2000; % no. of data points
m = 25; % no. of principal components
X = score(:,1:m);
%% Initialising mean for GMM

%u = c(:,1:m); % When kmeans centres are used for initialisation

ini_mean = randperm(n);% Randomly taking k data-points as means.
u = X(ini_mean(1:k), :);

%% Initialising variance for GMM

var = [];
for j = 1 : k,
    var{j} = cov(X);% Initialising covariance matrix by covariance of dataset as obtained after PCA
end

%% Prior probalities of clusters

ini_pi = (1 / k)*ones(1, k);

%% EM Algorithm

wts = zeros(n, k); %belongingness matrix

for itr = 1:500,
    %Calculate the objective function
    dist_fn = zeros(n, k);
    dist_fn_wt = zeros(n, k);

    for j = 1 : k
        temp = zeros(n,m);
        for i =1:n
            temp(i,:) = u(j,:);
        end
        mean_subtract = X-temp;
        %%var{j}=var{j}+eye(m)*0.000001;
        % calculating gaussian function for all data points for a cluster
        dist_fn(:,j) = (1 / sqrt((2*pi)^m * det(var{j}))) * exp(-1/2 * sum((mean_subtract * inv(var{j}) .* mean_subtract), 2));
        %%dist_fn(:,j) = log(dist_fn(:,j));
        % Multiply each pdf value by the prior probability for cluster.
        dist_fn_wt(:,j) = dist_fn(:,j).*ini_pi(j);
    end
   
    % Computing log derivative and posterior probabilities into weights
% %   temp1 = zeros(n,k);
% %     for j=1:k
% %         for j1=1:k
% %             temp1(:,j)=temp1(:,j)+ini_pi(j1)*exp(dist_fn(:,j1)-dist_fn(:,j));
% %         end
% %         for i=1:n
% %             wts(i,j) = ini_pi(j)/temp1(i,j);
% %         end
% %     end
     if (sum(dist_fn_wt,2)~=0)
          wts = bsxfun(@rdivide, dist_fn_wt, sum(dist_fn_wt, 2));
     end

    prv_u = u;        
    for j = 1 : k
        %% Updating means
        u(j, :) = wts(:,j)' * X;
        if (sum(wts(:,j),1)~=0)
            u(j,:) = u(j,:) ./ sum(wts(:,j), 1);
        end   
        %% update prior probability
        ini_pi(j) = mean(wts(:, j), 1);

        %% Updating variance 
        
        var_k = zeros(m, m);
        
        temp = zeros(n,m);
        for i =1:n
            temp(i,:) = u(j,:);
        end
        X_u = X - temp;
        
        for i = 1:n
            var_k = var_k + (wts(i, j) .* (X_u(i, :)' * X_u(i, :)));
        end
        
        if (sum(wts(:, j))~=0)
            var{j} = var_k ./ sum(wts(:, j));
        end   
%         %Dropping non diagonal entries to avoid singularity
% %         for i1=1:m
% %             for j1=1:m
% %                 if i1~=j1
% %                     var{j}(i1,j1)=0;
% %                 end
% %             end
% %         end
    end
    %% Check for convergence
    if (sum(abs(u-prv_u))<0.000001)
        break
    end
              
end
%% analysing results
labs = zeros(2000,1);
for i=1:2000
    [val,labs(i)]=max(wts(i,:));
end
%checking if point has significant posterior probability in more than one
%point
mixed_pts = zeros(2000,1);
for i=1:2000
    for j=1:k
        if wts(i,labs(i))-wts(i,j)<=0.5 && j~=labs(i)
            mixed_pts(i)=1;
        end
    end
end
ctr=0;
listpts = zeros(sum(mixed_pts),1);
for i=1:2000
    if mixed_pts(i)==1
        ctr=ctr+1;
        listpts(ctr)=i;
    end
end
%gathering cluster information
clusters = zeros(10,10);
for i=1:2000
    clusters(labs(i),t(i)+1)=clusters(labs(i),t(i)+1)+1;
end
%assigning cluster labels to most frequently occuring values
labels=int16([]);
for i=1:10
    labels(i)=0;
end
for i=1:10
    [val,lab]=max(clusters(i,:));
    labels(i)=lab-1;
end
%initial data labels
nums=int16([]);
for i=1:10
    nums(i)=0;
end
for i=1:2000
    nums(t(i)+1)=nums(t(i)+1)+1;
end
%calculating accuracy
acc = 0;
for i=1:2000
    if(labels(labs(i))==t(i))
        acc=acc+1;
    end
end
%percentage accuracy
perc = (acc/2000)*100;
%to assess clusters through visualisation, taking average image of
%containing data points in each cluster
data0=zeros(28,28,'double');
data1=zeros(28,28,'double');
data2=zeros(28,28,'double');
data3=zeros(28,28,'double');
data4=zeros(28,28,'double');
data5=zeros(28,28,'double');
data6=zeros(28,28,'double');
data7=zeros(28,28,'double');
data8=zeros(28,28,'double');
data9=zeros(28,28,'double');

for i=1:2000
    for j=1:28
        for k=1:28
            if (labs(i))==1
                data0(j,k)=(data0(j,k)+data_image(i,28*(j-1)+k));
            end
            if (labs(i))==2
                data1(j,k)=(data1(j,k)+data_image(i,28*(j-1)+k));
            end
            if (labs(i))==3
                data2(j,k)=(data2(j,k)+data_image(i,28*(j-1)+k));
            end
            if (labs(i))==4
                data3(j,k)=(data3(j,k)+data_image(i,28*(j-1)+k));
            end
            if (labs(i))==5
                data4(j,k)=(data4(j,k)+data_image(i,28*(j-1)+k));
            end
            if (labs(i))==6
                data5(j,k)=(data5(j,k)+data_image(i,28*(j-1)+k));
            end
            if (labs(i))==7
                data6(j,k)=(data6(j,k)+data_image(i,28*(j-1)+k));
            end
            if (labs(i))==8
                data7(j,k)=(data7(j,k)+data_image(i,28*(j-1)+k));
            end               
            if (labs(i))==9
                data8(j,k)=(data8(j,k)+data_image(i,28*(j-1)+k));
            end
            if (labs(i))==10
                data9(j,k)=(data9(j,k)+data_image(i,28*(j-1)+k));
            end
        end
    end
end
data0=data0/sum(clusters(1,:));
data1=data1/sum(clusters(2,:));
data2=data2/sum(clusters(3,:));
data3=data3/sum(clusters(4,:));
data4=data4/sum(clusters(5,:));
data5=data5/sum(clusters(6,:));
data6=data6/sum(clusters(7,:));
data7=data7/sum(clusters(8,:));
data8=data8/sum(clusters(9,:));
data9=data9/sum(clusters(10,:));