%% Extracting data from csv file
A = csvread('part1.csv');
x =  A(:,1);
t =  A(:,2);
xt = x(1:20);
tt = t(1:20);
x1 = x(1:16);
t1 = t(1:16);
x2 = x(17:20);
t2 = t(17:20);
%plot(x1,t1);
max_deg = 8;
tr_err = zeros(1,max_deg);
test_err = zeros(1,max_deg);
w_val = zeros(max_deg);
lambda=[1,10];
for i = 1:max_deg,
    options = optimoptions(@fminunc,'Algorithm','quasi-newton');
    [w,fval] = fminunc(@(w)quad_reg_fun(x1,t1,i,w,lambda(2)),zeros(1,i),options);
    tr_err(i)= fval;
    b = zeros(1,max_deg-i);
    w_val(i,:) = [w,b];
    test_err(i)= quad_reg_fun(x2,t2,i,w,lambda(2));
end
[min_err,pos]=min(test_err);  
plot(test_err,'o');
hold on
plot(tr_err,'b');
legend('test error','training error');
degree=pos-1;

figure('Name','Fitting Best Fit');
plot(xt,tt,'r');
hold on;
res=val(xt,w_val(pos,:),pos);
plot(xt(1:16),res(1:16),'o',xt(17:20),res(17:20),'--');
%legend('Actual Data','Predicted Data');

%Calculating variance

noise_var = sum((transpose(res)-tt).^2)/length(x1);