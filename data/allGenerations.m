% For all cases :

% Q_per - 0.15
% Q_range - 0.1
% random_load = true
% data_name = ndata.mat
% dc_ac = true
% G_range = 0
% L_range = 0
% Upper_bound = 1.2
% lower_bound = 0.8
% V_range = 0
% data_size - vary
% Va_range = 0
% ref = 0
% L_corr = 0

% DataGeneration('case_', 0.15, '_data.mat', true, 0, 1.2, 0.8, 0.1, 0, _, 0,true, 0,0,0); 
DataGeneration('case30', 0.15, '30data.mat', true, 0, 1.2, 0.8, 0.1, 0, 500, 0,true, 0,0,0);
DataGeneration('case39', 0.15, '39data.mat', true, 0, 1.2, 0.8, 0.1, 0, 600, 0,true, 0,0,0);
DataGeneration('case57', 0.15, '57data.mat', true, 0, 1.2, 0.8, 0.1, 0, 700, 0,true, 0,0,0);
DataGeneration('case118', 0.15, '118data.mat', true, 0, 1.2, 0.8, 0.1, 0, 800, 0,true, 0,0,0);
DataGeneration('case300', 0.15, '300data.mat', true, 0, 1.2, 0.8, 0.1, 0, 1000, 0,true, 0,0,0);
DataGeneration('case2383wp', 0.15, '2383wpdata.mat', true, 0, 1.2, 0.8, 0.1, 0, 3000, 0,true, 0,0,0);
DataGeneration('case2746wp', 0.15, '2746wpdata.mat', true, 0, 1.2, 0.8, 0.1, 0, 3500, 0,true, 0,0,0);
DataGeneration('case3375wp', 0.15, '3375wpdata.mat', true, 0, 1.2, 0.8, 0.1, 0, 4000, 0,true, 0,0,0);