%% Statistical Analysis of Spatial Room Impulse Responses (SRIRs)
% This script performs a comprehensive comparison of dimensionality reduction
% techniques—Principal Component Analysis (PCA), Linear Discriminant Analysis
% (LDA), and Exploratory Factor Analysis (EFA)—applied to acoustic parameters
% extracted from SRIR datasets. It evaluates classification performance using
% confusion matrices and accuracy metrics.
%
% The analysis includes:
% - Data loading and preprocessing
% - Feature standardization
% - Dimensionality reduction with PCA, LDA, and EFA
% - Train/test splitting and classification
% - Visualization of results via confusion matrices and accuracy bar plot
%
% Dependencies:
% - MATLAB Statistics and Machine Learning Toolbox
% - SDMtools (for acoustic parameter extraction; assumed in '../SDMtools/')
% - Custom functions (assumed in 'functions/')
%
% Dataset:
% - SRIR .mat files from Zenodo (DOI: 10.5281/zenodo.10450779)
% - Place .mat files in 'data/' folder
%
% Author: Muhammad Awais
% Contact: muhammadawais.de@gmail.com
% Date Created: [Original creation date]
% Last Updated: August 09, 2025

close all;
clear all;  % Clear all variables to start fresh
clc;        % Clear command window

%% Add Required Paths
% Add paths to external toolboxes and custom functions
addpath(genpath('../SDMtools/'));  % Path to Spatial Decomposition Method tools
addpath(genpath('functions/'));    % Path to custom functions

%% Setup Parameters
% Define sampling frequency and marker list for potential plotting
fs = 48000;  % Sampling frequency (Hz)
listMarker = ['o'; '+'; '*'; 'x'; 's'; '^'; 'v'; 'p'; 'h'; ...
              'o'; '+'; '*'; 'x'; 's'; '^'; 'v'; 'p'; 'h'];  % Markers for scatter plots

%% Load Dataset
% Load SRIR data from .mat files in the 'data/' directory if not already loaded
if ~exist('data', 'var')
    filepath = fullfile(pwd, 'data/');  % Construct full path to data folder
    Files = dir([filepath, '*.mat']);   % Get list of all .mat files
    data = [];                          % Initialize empty data array
    
    for k = 1:length(Files)
        disp([num2str(k), '/', num2str(length(Files))]);  % Display progress
        filename = Files(k).name;                         % Get current filename
        tmp = load([filepath, filename]);                 % Load the file
        data = [data; tmp.data];                          % Append to data array
    end
end

[lenData, ~] = size(data);  % Get number of data entries
roomsCol = data(:, 1);      % Extract room identifiers from first column

% Convert room identifiers to strings and find unique room names
roomsColStr = cellfun(@(x) string(x), roomsCol, 'UniformOutput', false);
roomsName = unique([roomsColStr{:}]);

%% Select Room Acoustic Parameters (RAP)
% Define selected octave bands and RAPs for analysis
select_band = 2:9;  % Octave bands from 62 Hz to 8 kHz (indices 2 to 9)

% Define RAP names, LaTeX labels, and band assignments
Para_select(1, :) = {'EDT', 'DT20m', 'D50', 'C50', 'C80', 'DRR', 'Grel', 'TS'};
Para_select(3, :) = {'EDT', 'DT_{20}', 'D_{50}', 'C_{50}', 'C_{80}', 'DRR', 'G_{rel}', 'T_{S}'};
Para_select(4, :) = num2cell(zeros(size(Para_select(1, :))));  % Initialize zeros for optional use

% Assign selected bands to each parameter
for idx = 1:length(Para_select(1, :))
    Para_select(2, idx) = {select_band};
end

% Define octave band labels
f0_oct_name_RAP = {'wB'; '62Hz'; '125Hz'; '250Hz'; '500Hz'; '1kHz'; '2kHz'; '4kHz'; '8kHz'; '16kHz'};
bandLabels = f0_oct_name_RAP(select_band);  % Selected band labels

% Create RAP mask using custom function (assumes create_RAP_Mask is defined in 'functions/')
[select_RAP, Para_select, label_RAP, ~] = create_RAP_Mask(data{1, 5}, Para_select, f0_oct_name_RAP);

% Extract RAP values from dataset
clear RAP;  % Clear any existing RAP variable
RAP = zeros(length(data), length(select_RAP));  % Preallocate RAP matrix
for idxK = 1:length(data)
    tmp = table2array(data{idxK, 5});            % Convert table to array
    tmp = cell2mat(tmp(select_RAP));             % Extract selected RAP values
    RAP(idxK, :) = tmp;                          % Store in RAP matrix
end

%% Assign Class Labels
% Map room names to numeric IDs for classification
roomID = zeros(lenData, 1);  % Preallocate room ID vector
for idxData = 1:length(data)
    roomID(idxData) = find(matches(roomsName, data(idxData, 1)));  % Find index of room name
end

Cl_select = roomID;    % Class labels (numeric room IDs)
Cl_names = roomsName;  % Class names (room strings)

%% Standardize Features
% Z-score normalization: subtract mean and divide by standard deviation
RAPmean = mean(RAP);             % Compute mean of each feature
RAPstd = std(RAP);               % Compute standard deviation of each feature
RAP_standard = (RAP - RAPmean) ./ RAPstd;  % Standardize the RAP matrix

%% Perform Principal Component Analysis (PCA)
% Reduce dimensionality while preserving 95% of variance
[W, PCA_LD, latent, ~, explained] = pca(RAP_standard, 'Algorithm', 'eig');  % Compute PCA
idxLD = find(cumsum(explained) >= 95, 1);  % Find number of components for 95% variance
PCA_features = PCA_LD(:, 1:idxLD);         % Extract selected PCA features

%% Perform Linear Discriminant Analysis (LDA)
% Supervised dimensionality reduction for class separation
lda_model = fitcdiscr(RAP_standard, Cl_select);  % Train LDA model
% Extract LDA features using the linear coefficients (assuming binary or multi-class setup)
LDA_features = RAP_standard * lda_model.Coeffs(1, 2).Linear;

%% Perform Exploratory Factor Analysis (EFA)
% Unsupervised latent factor extraction using Kaiser criterion
[~, D] = eig(cov(RAP_standard));                % Eigen decomposition of covariance matrix
eigenVals = flipud(sort(diag(D)));              % Sort eigenvalues descending
numFactors = sum(eigenVals > 1);                % Number of factors (eigenvalues > 1)
[Loadings, Psi, T, stats, F] = factoran(RAP_standard, numFactors, 'rotate', 'varimax');  % Perform EFA
EFA_features = F;                               % Extract EFA factor scores

%% Train/Test Split
% Stratified split with 70% training and 30% testing
rng(1);  % Set random seed for reproducibility
cv = cvpartition(Cl_select, 'HoldOut', 0.3);  % Create hold-out partition
trainIdx = training(cv);                      % Training indices
testIdx = test(cv);                           % Testing indices

%% Train Classifiers on Reduced Features
% PCA-based Classifier
pca_classifier = fitcdiscr(PCA_features(trainIdx, :), Cl_select(trainIdx));  % Train on PCA features
pca_preds = predict(pca_classifier, PCA_features(testIdx, :));               % Predict on test set

% LDA-based Classifier
lda_classifier = fitcdiscr(LDA_features(trainIdx, :), Cl_select(trainIdx));  % Train on LDA features
lda_preds = predict(lda_classifier, LDA_features(testIdx, :));               % Predict on test set

% EFA-based Classifier
efa_classifier = fitcdiscr(EFA_features(trainIdx, :), Cl_select(trainIdx));  % Train on EFA features
efa_preds = predict(efa_classifier, EFA_features(testIdx, :));               % Predict on test set

%% Generate Confusion Matrices
% Visualize classification performance for each method
figure;  % PCA Confusion Matrix
confusionchart(Cl_select(testIdx), pca_preds, ...
    'Title', 'PCA Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

figure;  % LDA Confusion Matrix
confusionchart(Cl_select(testIdx), lda_preds, ...
    'Title', 'LDA Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

figure;  % EFA Confusion Matrix
confusionchart(Cl_select(testIdx), efa_preds, ...
    'Title', 'EFA Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

%% Calculate Classification Accuracies
% Compute accuracy as percentage of correct predictions
pca_acc = sum(pca_preds == Cl_select(testIdx)) / length(pca_preds);  % PCA accuracy
lda_acc = sum(lda_preds == Cl_select(testIdx)) / length(lda_preds);  % LDA accuracy
efa_acc = sum(efa_preds == Cl_select(testIdx)) / length(efa_preds);  % EFA accuracy

% Display accuracies in command window
fprintf('\nClassification Accuracy Comparison:\n');
fprintf('PCA Accuracy: %.2f%%\n', pca_acc * 100);
fprintf('LDA Accuracy: %.2f%%\n', lda_acc * 100);
fprintf('EFA Accuracy: %.2f%%\n', efa_acc * 100);

%% Visualize Accuracy Comparison
% Bar plot for quick visual comparison of accuracies
figure;
bars = bar([pca_acc, lda_acc, efa_acc] * 100);  % Plot accuracies as percentages
set(gca, 'xticklabel', {'PCA', 'LDA', 'EFA'});  % Set x-axis labels
ylabel('Accuracy (%)');                        % Y-axis label
title('Classification Accuracy Comparison');    % Plot title
grid on;                                       % Add grid for readability