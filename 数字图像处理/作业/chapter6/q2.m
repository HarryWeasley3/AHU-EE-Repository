clc
clear
close all

% reading input image locations
% disp('Make sure the dimensions for both the images are the same');
% fg_img = input('Enter the full path to the foreground image: ', 's');
% bg_img = input('Enter the full path to the background image: ', 's');

% [filepath, name, ext] = fileparts(fg_img);

% % path to the output image
% output_img = strcat(filepath, '\', 'output.png');

% reading the images
fg = imread("img/fg.png");
bg = imread("img/bg.png");

% extracting image dimensions
size_img = size(fg);
w = size_img(1);
h = size_img(2);

% green pixel threshold
threshold = 255;

% initializing output array
output = zeros(w, h, 3, 'uint8');

for i = 1:1:w
    for j = 1:1:h
        % accessing the green pixel of fg image
        % 2 corresponds to green matrix
        g_pixel = fg(i, j, 2);

        if g_pixel >= threshold
            % if current pixel green value >= threshold
            % then select pixel from background
            output(i, j, 1) = bg(i, j, 1);               % 1 corresponds to red matrix
            output(i, j, 2) = bg(i, j, 2);               % 2 corresponds to green matrix
            output(i, j, 3) = bg(i, j, 3);               % 3 corresponds to blue matrix
        
        else
            % then select pixel from foreground
            output(i, j, 1) = fg(i, j, 1);
            output(i, j, 2) = fg(i, j, 2);
            output(i, j, 3) = fg(i, j, 3);
        end

    end
end

% imwrite(output, "img/output_img.png");
% disp(strcat('File - output.png', ' saved.'));
figure
imshow(output);
