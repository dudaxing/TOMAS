% run_homogenization_dataset.m
%
% Convenience wrapper that runs the Stokes-Brinkman fluid homogenization on
% the dataset produced by `scripts/run_generate_data.py`.
%
% USAGE (from MATLAB):
%   1. cd into the `dataset` folder.
%   2. >> run_homogenization_dataset
%
% By default this processes dataset number 1, reading
%   mstr_images_1.mat  -> mstr_images   (size: num_samples x nelx x nely)
% and writing
%   homogen_data_1.mat -> mstr, c00, c11, c01, c10  (each: num_samples x 1)
%
% Change `dataset_num` below to handle a different dataset.

clear all; close all; clc;

dataset_num = 1;
input_file  = sprintf('mstr_images_%d.mat',  dataset_num);
output_file = sprintf('homogen_data_%d.mat', dataset_num);

fprintf('Reading %s ...\n', input_file);
fprintf('Writing %s ...\n', output_file);

generate_homogenized_data(input_file, output_file);

fprintf('Done. You can now push %s back to the repo and run train_vae_main.ipynb.\n', output_file);
