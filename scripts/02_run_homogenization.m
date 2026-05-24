% Step 2: Periodic Stokes homogenization for each microstructure (MATLAB/Octave).
clear all; close all; clc;
script_dir = fileparts(mfilename('fullpath'));
dataset_dir = fullfile(script_dir, '..', 'dataset');
input_file = fullfile(dataset_dir, 'recons_shapes.mat');
output_file = fullfile(dataset_dir, 'homogen_data_1.mat');
addpath(dataset_dir);
generate_homogenized_data(input_file, output_file);
fprintf('Homogenization complete: %s\n', output_file);
