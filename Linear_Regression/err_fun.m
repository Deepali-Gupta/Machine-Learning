function Sum=err_fun(Phi,t,w,n,i)

Sum=0;
lambda=0;
for j=1:n
    V=0;
    for k=1:i
        V=V+(Phi(i,k)*w(k));
    end
    Sum = Sum + 0.5*(V-t(j))^2;
end
Sum = sqrt(2*Sum/length(t))+(lambda/2)*sum(abs(w));
%sum = sum/length(x);