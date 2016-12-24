function Sum=lasso_reg_fun(x,t,i,w,lambda)

Sum = 0;
for j=1:length(x)
   val = 0;
   for k = 1:i
     val = val + w(k)*(x(j)^(k-1));
   end
   Sum = Sum + 0.5*(val-t(j))^2;
   %sum = sum + abs(val-t(j));
end 
Sum = sqrt(2*Sum/length(x))+(lambda/2)*sum(abs(w));
%sum = sum/length(x);