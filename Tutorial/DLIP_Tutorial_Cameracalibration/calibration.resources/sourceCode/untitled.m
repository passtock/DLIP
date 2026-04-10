% Using Saved Camera Parameters
clear all; close all; clc;

load cameraParams.mat

% Load image
imageFileName = 'C:\Users\passp\source\repos\DLIP\Tutorial\DLIP_Tutorial_Cameracalibration\calibration.resources\chessboardIMGs\mdr0_20160719_085759_I.mp4_20160719_091046.703.jpg';
originalImage = imread(imageFileName);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% Display images (raw / undistorted)
figure, imshow(originalImage)
figure, imshow(undistortedImage)