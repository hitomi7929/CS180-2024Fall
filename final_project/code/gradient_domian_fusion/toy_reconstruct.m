function [im_out] = toy_reconstruct(im)

    [imh, imw, ~] = size(im);
    im2var = zeros(imh, imw);
    im2var(1:imh*imw) = 1:imh*imw;
    
    e = 0;
    N = imh * imw;
    A = sparse([], [], [], N, N);
    b = zeros(N, 1);
    
    % minimize (v(x+1,y)-v(x,y) - (s(x+1,y)-s(x,y)))^2
    for y = 1:imh
        for x = 1:imw-1
            e=e+1;
            A(e, im2var(y,x+1))=1;
            A(e, im2var(y,x))=-1;
            b(e) = im(y,x+1)-im(y,x);
        end
    end
    
    % minimize (v(x,y+1)-v(x,y) - (s(x,y+1)-s(x,y)))^2
    for y = 1:imh-1
        for x = 1:imw
            e=e+1;
            A(e, im2var(y+1,x))=1;
            A(e, im2var(y,x))=-1;
            b(e) = im(y+1,x)-im(y,x);
        end
    end
    
    % minimize (v(1,1)-s(1,1))^2	
    e=e+1;
    A(e, im2var(1,1))=1;
    b(e)=im(1,1);
    
    v = lscov(A, b);
    im_out = reshape(v, [imh, imw]);
    imshow(im_out)

end