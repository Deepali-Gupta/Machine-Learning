load('2013MT60079.mat');
%% setting labels in desired form for neural network computation
inputs = double(data_image');
targets = zeros(10,2000);
for i=1:2000
    targets(data_labels(i)+1,i)=1;
end
%% Create a Pattern Recognition Network
maxacc = 0;
index = 0;
x=zeros(10,1);
y=zeros(10,1);
z=zeros(10,1);
for i = 10:10
    
    hiddenSizes = [190 190];
    net = patternnet(hiddenSizes);
  
test1 = net.inputs{1}.processFcns;   %assigning the default value to some variable. You can see the options in the variable here.
test1 = test1(1,2);                               %changing its contents
net.inputs{1}.processFcns = test1;  %reassigning it.
%% Regularisation
% net.performFcn='mse';
% net.performParam.regularization= 0.8;

%% setting the learning rate
 net.layerWeights{ 2,1 }.learnParam.lr  = 0.01 ;
%% set activation function
net.layers{:}.transferFcn = 'logsig';
%% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%% Train the Network
[net,tr] = train(net,inputs,targets);
%  figure
%  plotperform(tr);

%% Testing the network
testX = inputs(:,tr.testInd);
testT = targets(:,tr.testInd);
trainX = inputs(:,tr.trainInd);
trainT = targets(:,tr.trainInd);

testY = net(testX);
testIndices = vec2ind(testY);
trainY = net(trainX);
trainIndices = vec2ind(trainY);
Y = net(inputs);

% figure
% plotconfusion(targets,Y);
[c,cm] = confusion(testT,testY);
[c2,cm2] = confusion(trainT,trainY);
[c3,cm3] = confusion(targets,Y);
perctest=100*(1-c);
perctrain=100*(1-c2);
perc=100*(1-c3);
% x(i/10)=i;
% y(i/10)=perctest;
% z(i/10)=perctrain;
if perctest>maxacc
    maxacc=perctest;
    index=i;
end
end
% figure
% plot(x,y,x,z);
% legend('Test','Train');
%fprintf('Percentage Correct Classification Test : %f%%\n', 100*(1-c));
%[c2,cm2] = confusion(trainT,trainY);
%fprintf('Percentage Correct Classification Train : %f%%\n', 100*(1-c2));
%[c3,cm3] = confusion(targets,Y);
%fprintf('Percentage Correct Classification Total : %f%%\n', 100*(1-c3));

 %plotroc(testT,testY)
% 
% 
%wb = getwb(net);
% w1=net.IW{1,1};
% bias = net.b{1};
% img = zeros(28);
% w = w1(10,:);
% tl = sum(w.^2);
% st = sqrt(tl);
% for i = 1:28
%     for j = 1:28
%         if((28*(j-1) + i)<=784)
%             img(i,j) = data_image(2000,(28*(i-1) + j));
%         end    
%     end
% end
% figure;
% imshow(img);
% wt=net.IW{1,1}
% coeff=zeros(784,80);
% for i=1:638
%     for j=1:80
%         coeff(i,j)=wt(j,i);
%     end
% end
% C=zeros(28,28,3);
% for i=1:28
%     for j=1:28
%         if 28*(i-1)+j <=638
%         if coeff(28*(i-1)+j,1)>0 & coeff(28*(i-1)+j,1)<1
%             C(i,j,1)=abs(coeff(28*(i-1)+j,1));
%         elseif coeff(28*(i-1)+j,1)<0 & coeff(28*(i-1)+j,1)<1
%              C(i,j,2)=abs(coeff(28*(i-1)+j,1));
%         else
%             C(i,j,:)=1;
%         end
%         end
%     end
% end