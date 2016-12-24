load('2013MT60079.mat');
M = double(data_image);
t = data_labels;
[coeff,score,latest,tsquared,explained] = pca(M);
%to plot residual variance
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

plot(x,resvar);
min=784;
for i=1:784
    if resvar(i)<=10
        min=i;
        break
    end
end
%to visualise important components
dotprod=zeros(5,10);
for i=1:5
    for i2=1:10
        dotprod(i,i2)=dot(mean(data_image((i2-1)*200+1:i2*200,:)),coeff(:,i));
    end
end
%apply kmeans in PCA space
N=score(:,1:83);
idx = kmeans(N,10,'Replicates',1);
%gathering cluster information
clusters = zeros(10,10);
for i=1:2000
    clusters(idx(i),t(i)+1)=clusters(idx(i),t(i)+1)+1;
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
    if(labels(idx(i))==t(i))
        acc=acc+1;
    end
end
%percentage accuracy
perc = (acc/2000)*100;
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
            if (idx(i))==1
                data0(j,k)=(data0(j,k)+data_image(i,28*(j-1)+k));
            end
            if (idx(i))==2
                data1(j,k)=(data1(j,k)+data_image(i,28*(j-1)+k));
            end
            if (idx(i))==3
                data2(j,k)=(data2(j,k)+data_image(i,28*(j-1)+k));
            end
            if (idx(i))==4
                data3(j,k)=(data3(j,k)+data_image(i,28*(j-1)+k));
            end
            if (idx(i))==5
                data4(j,k)=(data4(j,k)+data_image(i,28*(j-1)+k));
            end
            if (idx(i))==6
                data5(j,k)=(data5(j,k)+data_image(i,28*(j-1)+k));
            end
            if (idx(i))==7
                data6(j,k)=(data6(j,k)+data_image(i,28*(j-1)+k));
            end
            if (idx(i))==8
                data7(j,k)=(data7(j,k)+data_image(i,28*(j-1)+k));
            end               
            if (idx(i))==9
                data8(j,k)=(data8(j,k)+data_image(i,28*(j-1)+k));
            end
            if (idx(i))==10
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