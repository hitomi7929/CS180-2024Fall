function [im_gr] = color2gray(im_rgb)
    im_hsv = rgb2hsv(im_rgb);
    [imh, imw, ~] = size(im_hsv);
    N = imh*imw;
    im2var = zeros(imh, imw);
    im2var(1:N) = 1:N;

    e = 0;
    sparse_i = [];
    sparse_j = [];
    sparse_v = [];
    b = zeros(N, 1);

    for y = 1:imh
        for x = 1:imw
            e = e + 1;
            b(e) = 0;

            if y~=1
                sparse_i = [sparse_i e];
                sparse_j = [sparse_j im2var(y, x)];
                sparse_v = [sparse_v 1];
                sparse_i = [sparse_i e];
                sparse_j = [sparse_j im2var(y-1, x)];
                sparse_v = [sparse_v -1];
                b(e) = b(e) + im_hsv(y,x,2) - im_hsv(y-1,x,2);
            end
            if y~=imh
                sparse_i = [sparse_i e];
                sparse_j = [sparse_j im2var(y, x)];
                sparse_v = [sparse_v 1];
                sparse_i = [sparse_i e];
                sparse_j = [sparse_j im2var(y+1, x)];
                sparse_v = [sparse_v -1];
                b(e) = b(e) + im_hsv(y,x,2) - im_hsv(y+1,x,2);
            end
            if x~=imw
                sparse_i = [sparse_i e];
                sparse_j = [sparse_j im2var(y, x)];
                sparse_v = [sparse_v 1];
                sparse_i = [sparse_i e];
                sparse_j = [sparse_j im2var(y, x+1)];
                sparse_v = [sparse_v -1];
                b(e) = b(e) + im_hsv(y,x,2) - im_hsv(y,x+1,2);
            end
            if x~=1
                sparse_i = [sparse_i e];
                sparse_j = [sparse_j im2var(y, x)];
                sparse_v = [sparse_v 1];
                sparse_i = [sparse_i e];
                sparse_j = [sparse_j im2var(y, x-1)];
                sparse_v = [sparse_v -1];
                b(e) = b(e) + im_hsv(y,x,2) - im_hsv(y,x-1,2);
            end
            % b(e) = b(e) * 5;
        end
    end

    A = sparse(sparse_i, sparse_j, sparse_v, N, N);
    v = lscov(A, b);
    im_gr = reshape(v, [imh, imw]);
end