function error = err(x,t,w,i)
error = zeros(1,length(x));
for j=1:length(x)
   val = 0;
   for k = 1:i
     val = val + w(k)*(x(j,k));
   end
   error(j) =  abs(val-t(j));
end 
%sum = sqrt(2*sum/length(x));