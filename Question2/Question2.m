clear all;
close all;

% generate data from given program
[xTrain,yTrain,xValidate,yValidate] = A2Q2_data; 

nTrain = size(xTrain,2); %size of training data set
nValidate = size(xValidate,2); %size of training data set

% calcualte z(x) for training data
zTrain(:,:) = [ones(1, nTrain);
    xTrain(1,:);
    xTrain(2,:);
    xTrain(1,:).^2; 
    xTrain(1,:).*xTrain(2,:); 
    xTrain(2,:).^2; 
    xTrain(1,:).^3; 
    (xTrain(1,:).^2).*xTrain(2,:); 
    xTrain(1,:).*(xTrain(2,:).^2); 
    xTrain(2,:).^3;];

% calcualte z(x) for validation data 
zValidate(:,:) = [ones(1, nValidate); 
    xValidate(1,:);
    xValidate(2,:);
    xValidate(1,:).^2; 
    xValidate(1,:).*xValidate(2,:); 
    xValidate(2,:).^2; xValidate(1,:).^3; 
    (xValidate(1,:).^2).*xValidate(2,:); 
    xValidate(1,:).*(xValidate(2,:).^2); 
    xValidate(2,:).^3;];
%% Part A -ML Estimator
% calculate w_ML
for i = 1:nTrain
    A(:,:,i) = zTrain(:,i)*zTrain(:,i).'; 
    b(:,i) = zTrain(:,i)*yTrain(i);
end

w_ML = inv(sum(A,3))*sum(b,2);

% calculate mean squared error for w_ML
yValidateW_ML = w_ML.'*zValidate; % guessed points of validation dataset 
validateMean2errorML = (1/nValidate)*sum(((yValidate -yValidateW_ML).^2)); 
yTrainW_ML = w_ML.'*zTrain; % guessed points of training dataset 
trainMean2errorML = (1/nTrain)*sum(((yTrain -yTrainW_ML).^2));

% plot guessed points over actual points
yTrainW_ML = w_ML.'*zTrain; % guessed points of training dataset 
figure(1)
hold on
plot3(xTrain(1,:),xTrain(2,:),yTrainW_ML,'rx')
legend('Actual Data','MLE Data')
hold off
figure(2)
hold on
plot3(xValidate(1,:),xValidate(2,:),yValidateW_ML,'rx')
legend('Actual Data','MLE Data') 
hold off

%% Part B -MAP Estimator
% set values for sigma^2 and gamma
sigma2 = var(sqrt((yTrain -yTrainW_ML).^2)); % reasonable estimation 
gamma = logspace(-4,4,1000);

% calculate w_MAP and mean squared error for each w_MAP
for i = 1:length(gamma)
    clear A b
    for j = 1:nTrain
        A(:,:,j) = zTrain(:,j)*zTrain(:,j).' + (sigma2/(nTrain*gamma(i)))*eye(size(zTrain,1)); 
        b(:,j) = zTrain(:,j)*yTrain(j);
    end
    w_MAP(:,i) = inv(sum(A,3))*sum(b,2);
    yValidateW_MAP = w_MAP(:,i).'*zValidate; % guessed points of validation dataset 
    validateMean2errorMAP(i) = (1/nValidate)*sum(((yValidate -yValidateW_MAP).^2)); 
    yTrainW_MAP = w_MAP(:,i).'*zTrain; % guessed points of training dataset 
    trainMean2errorMAP(i) = (1/nTrain)*sum(((yTrain -yTrainW_MAP).^2));
end

%plot mean squared error wrt gamma
figure(3)
semilogx(gamma, validateMean2errorMAP, 'DisplayName', 'Validation Set Error') 
hold on
semilogx(gamma, trainMean2errorMAP, 'DisplayName', 'Training Set Error') 
title('MAP Mean Squared Error vs \gamma')
xlabel('\gamma')
ylabel('Mean Squared Error')

legend('Location','Southwest')
hold off

%Function Definitions
function [xTrain,yTrain,xValidate,yValidate] = A2Q2_data

Ntrain = 100; data = generateData(Ntrain);
figure(1), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), title('Training Dataset'),
xTrain = data(1:2,:); yTrain = data(3,:);

Nvalidate = 1000; data = generateData(Nvalidate);
figure(2), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), title('Validation Dataset'),
xValidate = data(1:2,:); yValidate = data(3,:);

end

function x = generateData(N)
gmmParameters.priors = [.3,.4,.3]; % priors should be a row vector
gmmParameters.meanVectors = [-10 0 10;0 0 0;10 0 -10];
gmmParameters.covMatrices(:,:,1) = [1 0 -3;0 1 0;-3 0 15];
gmmParameters.covMatrices(:,:,2) = [8 0 0;0 .5 0;0 0 .5];
gmmParameters.covMatrices(:,:,3) = [1 0 -3;0 1 0;-3 0 15];
[x,labels] = generateDataFromGMM(N,gmmParameters);
end

function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N); 
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds= [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end
