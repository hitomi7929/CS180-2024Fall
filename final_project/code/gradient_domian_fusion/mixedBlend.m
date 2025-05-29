function [im_blend] = mixedBlend(im_s, mask_s, im_background)

    [imh, imw, nb] = size(im_background);
    N = imh*imw;
    im2var = zeros(imh, imw);
    im2var(1:N) = 1:N;

    v_rgb = {};

    for c = 1:nb
        e = 0;
        sparse_i = [];
        sparse_j = [];
        sparse_v = [];
        b = zeros(N, 1);

        for y = 1:imh
            for x = 1:imw
                e = e + 1;
                b(e) = 0;
                if mask_s(y, x)
                    sparse_i = [sparse_i e];
                    sparse_j = [sparse_j im2var(y,x)];
                    sparse_v = [sparse_v 4];

                    % The neighbour above
                    if mask_s(y-1, x)
                        sparse_i = [sparse_i e];
                        sparse_j = [sparse_j im2var(y-1,x)];
                        sparse_v = [sparse_v -1];
                    else
                        b(e) = b(e) + im_background(y-1, x, c);
                    end

                    if abs(im_s(y, x, c) - im_s(y-1, x, c)) > abs(im_background(y, x, c) - im_background(y-1, x, c))
                        b(e) = b(e) + im_s(y, x, c) - im_s(y-1, x, c);
                    else
                        b(e) = b(e) + im_background(y, x, c) - im_background(y-1, x, c);
                    end

                    % The neighbour below
                    if mask_s(y+1, x)
                        sparse_i = [sparse_i e];
                        sparse_j = [sparse_j im2var(y+1,x)];
                        sparse_v = [sparse_v -1];
                    else
                        b(e) = b(e) + im_background(y+1, x, c);
                    end

                    if abs(im_s(y, x, c) - im_s(y+1, x, c)) > abs(im_background(y, x, c) - im_background(y+1, x, c))
                        b(e) = b(e) + im_s(y, x, c) - im_s(y+1, x, c);
                    else
                        b(e) = b(e) + im_background(y, x, c) - im_background(y+1, x, c);
                    end

                    % The left neighbour
                    if mask_s(y, x-1)
                        sparse_i = [sparse_i e];
                        sparse_j = [sparse_j im2var(y,x-1)];
                        sparse_v = [sparse_v -1];
                    else
                        b(e) = b(e) + im_background(y, x-1, c);
                    end

                    if abs(im_s(y, x, c) - im_s(y, x-1, c)) > abs(im_background(y, x, c) - im_background(y, x-1, c))
                        b(e) = b(e) + im_s(y, x, c) - im_s(y, x-1, c);
                    else
                        b(e) = b(e) + im_background(y, x, c) - im_background(y, x-1, c);
                    end

                    % The right neighbour
                    if mask_s(y, x+1)
                        sparse_i = [sparse_i e];
                        sparse_j = [sparse_j im2var(y,x+1)];
                        sparse_v = [sparse_v -1];
                    else
                        b(e) = b(e) + im_background(y, x+1, c);
                    end

                    if abs(im_s(y, x, c) - im_s(y, x+1, c)) > abs(im_background(y, x, c) - im_background(y, x+1, c))
                        b(e) = b(e) + im_s(y, x, c) - im_s(y, x+1, c);
                    else
                        b(e) = b(e) + im_background(y, x, c) - im_background(y, x+1, c);
                    end

                else
                    sparse_i = [sparse_i e];
                    sparse_j = [sparse_j im2var(y,x)];
                    sparse_v = [sparse_v 1];

                    b(e) = im_background(y, x, c);
                end
            end
        end

        A = sparse(sparse_i, sparse_j, sparse_v, N, N);
        v_rgb{c} = lscov(A, b);
    end

    for c = 1:nb
      im_blend(:,:,c) = reshape(v_rgb{c}, [imh imw]);
    end
    imshow(im_blend)

end