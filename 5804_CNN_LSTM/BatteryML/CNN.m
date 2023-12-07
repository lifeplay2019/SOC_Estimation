% Shane Hu
% CNN_LSTM SOC estimation
% Version: 2.23
% date: 2023.12.07
% charging profiles of V, I, T data
close all;
clear all;
clc

load B0005.mat
load B0006.mat
load B0007.mat
load B0018.mat

cap5 = extract_discharge(B0005);
cap6 = extract_discharge(B0006);
cap7 = extract_discharge(B0007);
cap18 = extract_discharge(B0018);

% figure
% plot(cap5), hold on, plot(cap6), plot(cap7)
% plot(0:180, 1.4*ones(1, 181),'k--','LineWidth', 2)
% hold off, grid on
% xlabel Cycle, ylabel Capacity(Ah)
% legend('Battery #5', 'Battery #6', 'Battery #7', 'Failure Threshold')
% title('Capacity Degradations in Cycle')

charInput5 = extract_charge_preprocessing(B0005);
charInput6 = extract_charge_preprocessing(B0006);
charInput7 = extract_charge_preprocessing(B0007);
charInput18 = extract_charge_preprocessing(B0018);

InitC5 = 1.86;
InitC6 = 2.04;
InitC7 = 1.89;
InitC18 = 1.86;

%   For better training since it retains the original distribution of data
%   except for a scaling factor and transforms all the data into the range of [0,1]:
[xB5, yB5, ym5, yr5] = minmax_norm(charInput5, InitC5, cap5);
[xB6, yB6, ym6, yr6] = minmax_norm(charInput6, InitC6, cap6);
[xB7, yB7, ym7, yr7] = minmax_norm(charInput7, InitC7, cap7);
[xB18, yB18, ym18, yr18] = minmax_norm(charInput18, InitC18, cap18);

% number to train
% BatteryNum = 5;
% BatteryNum = 6;
% BatteryNum = 7;
 BatteryNum = 18;
switch BatteryNum
    case 5
        Train_Input = xB5;
        Train_Output = yB5;
        yr = yr5;
        ym = ym5;
    case 6
        Train_Input = xB6;
        Train_Output = yB6;
        yr = yr6;
        ym = ym6;
    case 7
        Train_Input = xB7;
        Train_Output = yB7;
        yr = yr7;
        ym = ym7;
    case 18
        Train_Input = xB18;
        Train_Output = yB18;
        yr = yr18;
        ym = ym18;
end

% FNN with 10 hidden neurons
netFNN10 = feedforwardnet(10);
netFNN10.trainParam.epochs = 300;
[netFNN10, tr] = train(netFNN10, Train_Input', Train_Output', 'useparallel',  'yes');

netFNN40 = feedforwardnet(40);
netFNN40.divideFcn = 'divideind';
netFNN40.divideParam.trainInd = tr.trainInd;
netFNN40.divideParam.valInd = tr.valInd;
netFNN40.divideParam.testInd = tr.testInd;
netFNN40.trainParam.epochs = 300;
netFNN40 = train(netFNN40, Train_Input', Train_Output', 'useparallel', 'yes');

% CNN1 : 2 Convolution Layer with filter size [1, 2] and number of filter 10, 5.
layerCNN1 = [
    imageInputLayer([1, 30]);
    convolution2dLayer([1, 2], 10, 'Stride', 1);
    leakyReluLayer
    convolution2dLayer([1, 2], 5, 'Stride', 1);
    leakyReluLayer
    fullyConnectedLayer(1)
    regressionLayer();
    ];
cellx = num2cell(Train_Input', 1)';
cellx = cellfun(@transpose, cellx, 'UniformOutput', false);
cellyB = num2cell(Train_Output);
tbl = table(cellx);
tbl.cellyB = cellyB;

Traintbl = tbl(tr.trainInd, :);
valtbl = tbl(tr.valInd, :);
testtbl = tbl(tr.testInd, :);

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs',500, ...
    'MiniBatchSize',50, ...
    'Plots','training-progress', 'ValidationData', valtbl);

netCNN1 = trainNetwork(Traintbl, layerCNN1, options);

layerCNN2 = [
    imageInputLayer([1, 30]);
    convolution2dLayer([1, 2], 30, 'Stride', 1);
    leakyReluLayer
    convolution2dLayer([1, 2], 15, 'Stride', 1);
    leakyReluLayer
    fullyConnectedLayer(1)
    regressionLayer();
    ];
netCNN2 = trainNetwork(Traintbl, layerCNN2, options);

inputSize = 30;
numHiddenUnits = 50;

layersLSTM = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(1)
    regressionLayer
    ];

cellx = num2cell(Train_Input', 1)';
cellyB = num2cell(Train_Output);

traincellx = cellx(tr.trainInd, :);
valcellx = cellx(tr.valInd, :);
testcellx = cellx(tr.testInd, :);

traincellyB = cellyB(tr.trainInd, :);
valcellyB = cellyB(tr.valInd, :);
testcellyB = cellyB(tr.testInd, :);

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs',500, ...
    'MiniBatchSize',50, ...
    'Plots','training-progress', 'ValidationData', {valcellx, valcellyB});

netLSTM = trainNetwork(traincellx, traincellyB, layersLSTM, options);

pFNN10 = netFNN10(Train_Input(tr.testInd, :)');
pFNN40 = netFNN40(Train_Input(tr.testInd, :)');
cellx = num2cell(Train_Input(tr.testInd, :)', 1)';
cellx = cellfun(@transpose, cellx, 'UniformOutput', false);
tbl = table(cellx);

x_4d = zeros(1, 30, 1, height(tbl));
for i = 1:height(tbl)
    x_4d(:,:,:,i) = tbl.cellx{i};
end
pCNN1 = predict(netCNN1, x_4d);

pCNN2 = predict(netCNN2, x_4d);

pLSTM = cell2mat(predict(netLSTM, num2cell(Train_Input(tr.testInd, :)', 1)));

Train_Output = Train_Output(tr.testInd, :)*yr + ym;
pFNN10 = pFNN10*yr+ ym;
pFNN40 = pFNN40*yr + ym;
pCNN1 = pCNN1*yr + ym;
pCNN2 = pCNN2*yr + ym;
pLSTM = pLSTM*yr + ym;
% figure, hold on, grid on,
% plot(Train_Output, 'linewidth', 2), plot(pLSTM, '-.')
% plot(1:25, 1.4*ones(1, 25),'k--','LineWidth', 2), xlim([1 25])
% title(['Capacity Estimation using V, I, T (Battery #', num2str(BatteryNum), ')'])
% xlabel Cycle, ylabel Capacity(Ah)
% legend('Real Value', 'LSTM Predicted', 'Failure Threshold')

