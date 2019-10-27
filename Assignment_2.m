%% Initialization
clear ; close all;clc

% Load Data

ch=input("Choice: 1.Iris 2.Divorce Predictors(enter 1 or 2)");

if ch == 1
  load('iris.mat');
else
  load('divorce.mat');
end

% Training_set:Test_set=6:4

input_layer_size  = size(x_train,2);     
num_labels = max(y_train)-min(y_train)+1;

hidden_layer_size = floor((input_layer_size - num_labels)/2.5);
if hidden_layer_size < num_labels
  hidden_layer_size = num_labels+4; 
end

% 3 layers   

v = rand(hidden_layer_size, input_layer_size);
w = rand(num_labels, hidden_layer_size);
r = rand(hidden_layer_size, 1);
t = rand(num_labels, 1);
a1 = 0.1;
a2 = 0.1;

m = length(y_train);

for i = 1: m
	yk(y_train(i)+1,i)=1;
end

j = 0;

while j < 800
  
  for i = 1: m
    
    b = sigmoid(v*x_train(i, :)'-r);
    y = sigmoid(w*b-t);
    g = y.*(1-y).*(yk(:,i)-y);
    e = b.*(1-b).*sum(w.*g)';
    w = w+a1*g*b';
    t = t-a1*g;
    v = v+a2*e*x_train(i, :);
    r = r-a2*e;
    
  end
  
  j = j+1;
end

fprintf('\nTraining Set Accuracy: %f\n', Accuracy(x_test,y_test,v,w,r,t,num_labels));
