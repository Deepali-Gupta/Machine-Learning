%% Extracting data
%fminsearch
%basis function sigmoidal

n=3300;
M = csvread('train.csv',0,0,[0 0 n-1 18]);
M2= csvread('train.csv',n,0);
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
x21 = M2(:,1);
x22 = M2(:,2);
x23 = M2(:,3);
x24 = M2(:,4);
x25 = M2(:,5);
x26 = M2(:,6);
x27 = M2(:,7);
x28 = M2(:,8);
x29 = M2(:,9);
x210 = M2(:,10);
x211 = M2(:,11);
x212 = M2(:,12);
x213 = M2(:,13);
x214 = M2(:,14);
x215 = M2(:,15);
x216 = M2(:,16);
x217 = M2(:,17);
x218 = M2(:,18);
t2 = M2(:,19);

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
x = [ones(n,1),x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18];
x2 = [ones(3369-n,1),x21,x22,x23,x24,x25,x26,x27,x28,x29,x210,x211,x212,x213,x214,x215,x216,x217,x218];
xt = [ones(3369,1),xt1,xt2,xt3,xt4,xt5,xt6,xt7,xt8,xt9,xt10,xt11,xt12,xt13,xt14,xt15,xt16,xt17,xt18];
nx = [ones(706,1),nx1,nx2,nx3,nx4,nx5,nx6,nx7,nx8,nx9,nx10,nx11,nx12,nx13,nx14,nx15,nx16,nx17,nx18];
%% linear regression on data and seeing coefficients
w = zeros(19);%parameter vector
Phi = zeros(n,19);
for i=1:n
    
    for j=1:19
        Phi(i,j)=(x(i,j)^(1));
    end
end
%w = pinv(transpose(Phi)*Phi)*transpose(Phi)*t;
w=lasso(Phi,t);

%w_val = zeros(19);
i=19;

    options = optimoptions(@fminunc,'Algorithm','quasi-newton');
    [w,fval] = fminunc(@(w)err_fun(Phi,t,w,n,i),zeros(1,i),options);
    tr_err= fval;
w=transpose(w);
y=x*w;
yt=xt*w;
y2=x2*w;
tr_err=sum((y-t).^2)/length(t);
test_err=sum((y2-t2).^2)/length(t2);
res=nx*w;
csvwrite('result.csv',res);
plot(tt,'r');
hold on;
plot(yt,'b');
legend('true value','predicted value');
err=sum((yt-tt).^2)/length(tt);
