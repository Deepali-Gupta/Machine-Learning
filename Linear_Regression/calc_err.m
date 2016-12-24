function sum = calc_err(x,t,w,i)
sum = 0;
for j=1:length(x)
   val = 0;
   for k = 1:i
     val = val + w(k)*(x(j,k));
   end
   sum = sum + 0.5*(val-t(j))^2;
end 
sum = sqrt(2*sum/length(x));