%% Extracting data from csv file
A = csvread('part1.csv');
x =  A(:,1);
t =  A(:,2);
x1 = x(1:20);
t1 = t(1:20);
plot(x1,t1);
max_deg = 20;
tr_err = zeros(1,max_deg);
w_val = zeros(max_deg);

for i = 1:max_deg,
    options = optimoptions(@fminunc,'Algorithm','quasi-newton');
    [w,fval] = fminunc(@(w)err_fun(x1,t1,w,i),zeros(1,i),options);
    tr_err(i)= fval;
    b = zeros(1,max_deg-i);
    w_val(i,:) = [w,b];
end
[min_err,pos]=min(tr_err);  
degree=pos-1;

figure('Name','Fitting Best Fit');
plot(x1,t1);
hold on;
res=val(x1,w_val(pos,:),pos);
plot(x1,res,'o');
legend('Actual Data','Predicted Data');

%Calculating variance

noise_var = sum((transpose(res)-t1).^2)/length(x1);
