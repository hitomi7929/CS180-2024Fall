% starter script for project 3
DO_TOY = true;
DO_BLEND = true;
DO_MIXED  = true;
DO_COLOR2GRAY = true;

% Create the folder to save the outputs
mkdir('./output');

if DO_TOY 
    toyim = im2double(imread('./samples/toy_problem.png')); 
    % im_out should be approximately the same as toyim
    im_out = toy_reconstruct(toyim);
    imwrite(im_out, './output/toy_reconstruction.png');
    disp(['Error: ' num2str(sqrt(sum(toyim(:)-im_out(:))))])
end

if DO_BLEND
    % The first blended image
    im_background = imresize(im2double(imread('./samples/im2.jpg')), 0.5, 'bilinear');
    im_object = imresize(im2double(imread('./samples/penguin-chick.jpeg')), 0.5, 'bilinear');

    % get source region mask from the user
    objmask = getMask(im_object);
    % align im_s and mask_s with im_background
    [im_s, mask_s] = alignSource(im_object, objmask, im_background, true);
    imwrite(mask_s, './output/mask1.png');

    % blend
    im_blend = poissonBlend(im_s, mask_s, im_background);
    imwrite(im_blend, './output/blend1.png');
    figure(3), hold off, imshow(im_blend)

    % The second blended image
    im_background = imresize(im2double(imread('./samples/polarbear.jpg')), 0.8, 'bilinear');
    im_object = imresize(im2double(imread('./samples/penguin.jpg')), 0.35, 'bilinear');

    % get source region mask from the user
    objmask = getMask(im_object);
    % align im_s and mask_s with im_background
    [im_s, mask_s] = alignSource(im_object, objmask, im_background, false);

    % blend
    im_blend = poissonBlend(im_s, mask_s, im_background);
    imwrite(im_blend, './output/blend2.png');
    figure(3), hold off, imshow(im_blend)

    % The third blended image
    im_background = imresize(im2double(imread('./samples/plane.jpg')), 0.8, 'bilinear');
    im_object = imresize(im2double(imread('./samples/balloon.jpg')), 0.35, 'bilinear');

    % get source region mask from the user
    objmask = getMask(im_object);
    % align im_s and mask_s with im_background
    [im_s, mask_s] = alignSource(im_object, objmask, im_background, false);

    % blend
    im_blend = poissonBlend(im_s, mask_s, im_background);
    imwrite(im_blend, './output/blend3.png');
    figure(3), hold off, imshow(im_blend)

    % The fourth blended image
    im_background = imresize(im2double(imread('./samples/spaceship.jpg')), 0.8, 'bilinear');
    im_object = imresize(im2double(imread('./samples/balloon.jpg')), 0.5, 'bilinear');

    % get source region mask from the user
    objmask = getMask(im_object);
    % align im_s and mask_s with im_background
    [im_s, mask_s] = alignSource(im_object, objmask, im_background, false);
    
    % blend
    im_blend = poissonBlend(im_s, mask_s, im_background);
    imwrite(im_blend, './output/blend4.png');
    figure(3), hold off, imshow(im_blend)
end

if DO_MIXED
    % The first mixed blended image
    % read images
    im_bg = imresize(im2double(imread('./samples/im2.jpg')), 0.5, 'bilinear');
    im_object = imresize(im2double(imread('./samples/penguin-chick.jpeg')), 0.5, 'bilinear');

    % get source region mask from the user
    objmask = getMask(im_object);
    % align im_s and mask_s with im_background
    [im_s, mask_s] = alignSource(im_object, objmask, im_bg, false);
    
    % blend
    im_blend = mixedBlend(im_s, mask_s, im_bg);
    imwrite(im_blend, './output/mixed_blend1.png');
    figure(3), hold off, imshow(im_blend);

    % The second mixed blended image
    % read images
    im_bg = imresize(im2double(imread('./samples/plane.jpg')), 0.8, 'bilinear');
    im_object = imresize(im2double(imread('./samples/balloon.jpg')), 0.35, 'bilinear');

    % get source region mask from the user
    objmask = getMask(im_object);
    % align im_s and mask_s with im_background
    [im_s, mask_s] = alignSource(im_object, objmask, im_bg, false);
    
    % blend
    im_blend = mixedBlend(im_s, mask_s, im_bg);
    imwrite(im_blend, './output/mixed_blend2.png');
    figure(3), hold off, imshow(im_blend);
end

if DO_COLOR2GRAY
    im_rgb = im2double(imread('./samples/colorBlindTest35.png'));
    imshow(im_rgb)
    im_gr = color2gray(im_rgb);
    imwrite(im_gr, './output/color2gray.png');
    figure(4), hold off, imagesc(im_gr), axis image, colormap gray
end
