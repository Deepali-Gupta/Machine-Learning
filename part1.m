%file reading
N=20;
M=csvread('part1.csv');
xt=M(:,1);
tt=M(:,2);
x=xt(1:20);
t=tt(1:20);
err=zeros(1,20);
plot(xt,tt,'o');
hold on;
plot(x,t,'--');
%% polyfit without regularization
% for i=1:3,
%     p=polyfit(x,t,i);
%     s=polyval(p,x);
%     err(i)=0.5*sum((s-t).^2);
% end
% [min_err,I]=min(err);
Poly=polyfit(x,t,10);

hold on;
plot(xt,polyval(Poly,xt),'r');

