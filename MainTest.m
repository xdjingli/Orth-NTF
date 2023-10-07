clear all
clc

addpath([pwd, '/funs']);
addpath([pwd, '/datasets']);

%% load Dataset
datasetName = 'MSRC';
load([datasetName, '.mat']);

gt = Y;
num_Cluster = length(unique(gt));               
num_V = length(X);                              
num_N = size(X{1},1);                           
 
%% Data preprocessing
% Select a data preprocessing method, or no data preprocessing
% MSRC:     None 
% HW1256:   Data pre-processing B
% Mnist4:   Data pre-processing A
% AWA:      Data pre-processing A
%% Data pre-processing A
% disp('------Data preprocessing------');
% tic
% for v=1:num_V
%     a = max(X{v}(:));
%     X{v} = double(X{v}./a);
% ends
% toc

%% Data pre-processing B
% disp('------Data preprocessing------');
% tic
% for v=1:num_V
%     XX = X{v};
% for n=1:size(XX,1)
%     XX(n,:) = XX(n,:)./norm(XX(n,:),'fro');
% end
% X{v} = double(XX);
% end
% toc


%% parameter
% MSRC:     anchorRate:0.7 p:0.5 lambda:100
% HW1256:   anchorRate:1.0 p:0.1 lambda:1180
% Mnist4:   anchorRate:0.6 p:0.1 lambda:5000
% AWA:      anchorRate:1.0 p:0.5 lambda:1000
anchorRate = [0.7];
p = [0.5];
lambda = [100];

anchorNum = fix(num_N * anchorRate);

for num1 = 1:length(anchorNum)
    fprintf('------Current Anchor number:%d------\n', anchorNum(num1));

    %% result file
    dir_name = ['.\result\', datasetName, '\'];
    file_dir = [dir_name, datasetName, '_with_', int2str(anchorNum(num1)), 'AnchorPoints.txt'];
    if ~isfolder(dir_name)
        mkdir(dir_name);
    end
    fid = fopen(file_dir,'a');  

    %% 
    disp('----------Anchor Selection----------');
    tic;
    opt1.style = 1;          
    opt1.IterMax = 50;                      
    opt1.toy = 0;

    [~, B_init] = FastmultiCLR(X, num_Cluster, anchorNum(num1), opt1, 10);    
    toc;

    %% 
    B_init_hat = time2frequency(B_init);
    for v = 1:num_V 
        F_init_hat{v} = eye(num_N, num_Cluster);
        G_init_hat{v} = B_init_hat{v}' * F_init_hat{v};
    end 

    %% 
    for num2 = 1:length(p)
        for num3 = 1:length(lambda)
            [F, alpha] = OrthNTF(num_N, num_V, num_Cluster, B_init_hat, F_init_hat, G_init_hat, p(num2), lambda(num3));

            F_sum = F{1} / alpha(1);
            for v = 2:num_V
                F_sum = F_sum + F{v} / alpha(v);
            end
            alpha_sum = sum(1 ./ alpha);
            F_final = F_sum / alpha_sum;

            disp('----------Clustering----------');
            [~, Y_pre] = max(F_final, [], 2); 

            my_result = ClusteringMeasure1(Y, Y_pre);
            my_result

            fprintf(fid, 'P:%f ', p(num2));
            fprintf(fid, 'lambda:%f ', lambda(num3));
            fprintf(fid, '%g %g %g %g %g %g %g \n',my_result');
        end
    end
end

fclose(fid);