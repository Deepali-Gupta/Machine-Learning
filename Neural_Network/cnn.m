
%%  Load the MNIST dataset
addpath(genpath('../util'));
addpath(genpath('../CNN'));
% Load the MNIST hand-written digit dataset.
load mnist_uint8;
% Reshape the example digits back into 2D images.
%
% 'train_x' and 'test_x' begin as 2D matrices with one image per row.
%   train_x  [60000  x  784]
%   test_x   [10000  x  784]
%
% Also, rescale the pixel values from 0 - 255 to 0 - 1. 
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

colormap gray;
imagesc(train_x(:, :, 105)');
axis square;

%%  Define and Train the CNN

% Define the architecture of our CNN.
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 4, 'kernelsize', 9) %convolution layer
    struct('type', 's', 'scale', 4) %sub sampling layer
};

% Reset the seed generator. This will make your results reproducible.
rand('state', 0)

% Options for training.
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 100;

% Create all of the parameters for the network and randomly initialize
% them.
cnn = cnnsetup(cnn, train_x, train_y);

fprintf('Training the CNN...\n');

startTime = tic();

% Train the CNN using the training data.
cnn = cnntrain(cnn, train_x, train_y, opts);

fprintf('...Done. Training took %.2f seconds\n', toc(startTime));

%%  Test the CNN on the test set

fprintf('Evaluating test set...\n');

% Evaluate the trained CNN over the test samples.
[er, bad] = cnntest(cnn, test_x, test_y);

% Calculate the number of correctly classified examples.
numRight = size(test_y, 2) - numel(bad);

fprintf('Accuracy: %.2f%%\n', numRight / size(test_y, 2) * 100); 

% Plot mean squared error over the course of the training.
figure(1); 
plot(cnn.rL);
title('Mean Squared Error');
xlabel('Training Batch');
ylabel('Mean Squared Error');
%% visualising the weights of various layers
wt1=cnn.layers{1}.a{1,1};
wt2=zeros(28,28);
for i=1:50
    wt2=wt2+wt1(:,:,i);
end
wt1=wt2./50;
wt2=cnn.layers{2}.a{1};
wt3=zeros(20,20);
for i=1:50
    wt3=wt3+wt2(:,:,i);
end
wt2=wt3./50;
wt3=cnn.layers{3}.a{1};
wt4=zeros(5,5);
for i=1:50
    wt4=wt4+wt3(:,:,i);
end
wt3=wt4./50;
