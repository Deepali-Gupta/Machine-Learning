function V = val(x,w,i)
V=zeros(1,length(x));
for j=1:length(x),
    for k=1:i,
        V(j)=V(j)+(x(j,k)*w(k));
    end
end