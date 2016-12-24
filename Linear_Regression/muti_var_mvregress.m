%% Extracting data
M = csvread('train.csv',0,0,[0 0 1999 18]);
Mt = csvread('train.csv');
N = csvread('test.csv');
x1 = M(:,1);
x2 = M(:,2);
x3 = M(:,3);
x4 = M(:,4);
x5 = M(:,5);
x6 = M(:,6);
x7 = M(:,7);
x8 = M(:,8);
x9 = M(:,9);
x10 = M(:,10);
x11 = M(:,11);
x12 = M(:,12);
x13 = M(:,13);
x14 = M(:,14);
x15 = M(:,15);
x16 = M(:,16);
x17 = M(:,17);
x18 = M(:,18);
t = M(:,19);
xt1 = Mt(:,1);
xt2 = Mt(:,2);
xt3 = Mt(:,3);
xt4 = Mt(:,4);
xt5 = Mt(:,5);
xt6 = Mt(:,6);
xt7 = Mt(:,7);
xt8 = Mt(:,8);
xt9 = Mt(:,9);
xt10 = Mt(:,10);
xt11 = Mt(:,11);
xt12 = Mt(:,12);
xt13 = Mt(:,13);
xt14 = Mt(:,14);
xt15 = Mt(:,15);
xt16 = Mt(:,16);
xt17 = Mt(:,17);
xt18 = Mt(:,18);
tt = Mt(:,19);
nx1 = N(:,1);
nx2 = N(:,2);
nx3 = N(:,3);
nx4 = N(:,4);
nx5 = N(:,5);
nx6 = N(:,6);
nx7 = N(:,7);
nx8 = N(:,8);
nx9 = N(:,9);
nx10 = N(:,10);
nx11 = N(:,11);
nx12 = N(:,12);
nx13 = N(:,13);
nx14 = N(:,14);
nx15 = N(:,15);
nx16 = N(:,16);
nx17 = N(:,17);
nx18 = N(:,18);
x = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18];
xt = [xt1,xt2,xt3,xt4,xt5,xt6,xt7,xt8,xt9,xt10,xt11,xt12,xt13,xt14,xt15,xt16,xt17,xt18];
nx = [nx1,nx2,nx3,nx4,nx5,nx6,nx7,nx8,nx9,nx10,nx11,nx12,nx13,nx14,nx15,nx16,nx17,nx18];
%% linear regression on data and seeing coefficients
D = x2fx(x,'linear');
Dt = x2fx(xt,'linear');
nD = x2fx(nx,'linear');
[beta,sigma,e1,cov,log] = mvregress(D,t);
y = D*beta;
yt = Dt*beta;
ny = nD*beta;
csvwrite('result.csv',ny);
plot(yt,'r');
hold on;
plot(tt,'b');
legend('predicted value','true value');

abs_error = sum(abs(e1))/length(x1);
sq_error = sum(e1.^2)/length(x1);
rms_error  = sqrt(2*sum(e1.^2)/length(x1));
mse=sum((tt-yt).^2)/length(xt1);
print rms_error