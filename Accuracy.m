function A=Accuracy(x,y,v,w,r,t,num_labels)

m = length(y);
yk = zeros(num_labels,m);
j = zeros(m,1);

for i = 1: m
    
    yk(y(i)+1,i) = 1;
    yp = sigmoid(w*sigmoid(v*x(i, :)'-r)-t);
    
    if sum(abs(yp-yk(:,i))) < 0.1*size(yk,1)
        j(i)=1;
    end
    
end

A = 100*sum(j)/m;

end