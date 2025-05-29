import numpy as np
import skimage.io as skio
import skimage as sk
import scipy.interpolate
import cv2
from scipy.ndimage import distance_transform_edt


# Save the normalized image.
def save_normalized_image(image, imname):
    im_normalized = (image - image.min()) / (image.max() - image.min())
    im_uint8 = (im_normalized * 255).astype(np.uint8)
    fname = f'./output/{imname}.jpg'
    skio.imsave(fname, im_uint8)

# Compute the homography matrix.
def computeH(im1_pts, im2_pts):
    assert im1_pts.shape == im2_pts.shape, "Point sets must have the same shape"
    assert im1_pts.shape[0] >= 4 and im2_pts.shape[0] >= 4, "At least 4 points are required"
    A = []
    b = []
    rows = len(im1_pts)
    for i in range(rows):
        x1, y1 = im1_pts[i]
        x2, y2 = im2_pts[i]
        A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        b.append([x2])
        b.append([y2])
    A = np.array(A)
    b = np.array(b)
    h = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(b)
    H = np.array([[h[0, 0], h[1, 0], h[2, 0]],
                  [h[3, 0], h[4, 0], h[5, 0]],
                  [h[6, 0], h[7, 0], 1.0]])
    return H

# Add borders in order to present the whole image.
def add_border(im, im_pts, ratio_h, ratio_w):
    h, w, _ = im.shape
    h_add, w_add = int(h * ratio_h), int(w * ratio_w)
    im_border = np.zeros((h + 2 * h_add, w + 2 * w_add, im.shape[2]), dtype=im.dtype)
    im_border[h_add : h_add + h, w_add : w_add + w] = im
    im_pts += np.array([w_add, h_add])
    return im_border, im_pts

# Process the inverse warping.
def warpImage(im, H):
    height, width, _ = im.shape

    y_out, x_out = np.indices((height, width))
    coords = np.stack([x_out.ravel(), y_out.ravel(), np.ones_like(x_out).ravel()])

    H_inv = np.linalg.inv(H)
    warped_coords = H_inv @ coords
    warped_coords /= warped_coords[2]
    warped_coords = warped_coords[:2].T

    im_warped = np.zeros_like(im)
    for i in range(3):
        im_warped[..., i] = scipy.interpolate.griddata(
            warped_coords, im[..., i].ravel(),
            (x_out, y_out), method='nearest', fill_value=0
        )

    return im_warped

# Perform warping to align the source image to the target image.
def warping(im1, im2, im1_points, im2_points, ratio_h, ratio_w):
    im1_border, im1_pts = add_border(im1, im1_points, ratio_h, ratio_w)
    im2_border, im2_pts = add_border(im2, im2_points, ratio_h, ratio_w)

    H = computeH(im2_pts, im1_pts)
    return warpImage(im1_border, H), im2_border

# Compute the distance transform.
def distance(im):
    non_black_mask = np.any(im != [0, 0, 0], axis=-1).astype(np.float32)
    dist = distance_transform_edt(non_black_mask)
    max_dist = np.max(dist)
    dist = dist / max_dist
    dist = np.stack([dist, dist, dist], axis=-1)
    return dist

# Create the alpha mask.
def create_alpha_mask(im1, im2, kernel_size=(3,3), sigma=1):
    alpha = (distance(im1) > distance(im2)).astype(np.float32)
    return cv2.GaussianBlur(alpha, kernel_size, sigma)

# Generate a Gaussian stack.
def build_gaussian_stack(image, kernel_size, sigma, level):
    stack = []
    stack.append(image)
    for _ in range(level):
        stack.append(cv2.GaussianBlur(stack[-1], kernel_size, sigma))
    return stack

# Generate a Laplacian stack from the given Gaussian stack.
def build_laplacian_stack(gaussian_stack):
    laplacian_stack = []
    N = len(gaussian_stack) - 1
    for i in range(N):
        laplacian_stack.append(gaussian_stack[i] - gaussian_stack[i + 1])
    laplacian_stack.append(gaussian_stack[N])
    return laplacian_stack

# Blend two resource images together.
def blend(laplacian_stack1, laplacian_stack2, mask_gaussian_stack, shape):
    blend_stack = []
    for l1, l2, m in zip(laplacian_stack1, laplacian_stack2, mask_gaussian_stack):
        blend_stack.append(l1 * (1 - m) + l2 * m)
    blend_img = np.zeros(shape)
    reverse_stack = reversed(blend_stack)
    for laplacian in reverse_stack:
        blend_img = laplacian + blend_img
    return (blend_img - blend_img.min()) / (blend_img.max() - blend_img.min())

if __name__ == "__main__":
    ''' 
        Manually create mosaic for model.
    '''

    imname1 = './media/model_left.jpg'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = './media/model_center.jpg'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im1_points = np.loadtxt('./media/pts_model_left_lc.txt')
    im2_points = np.loadtxt('./media/pts_model_center_lc.txt')

    # Set the ratio of the borders.
    ratio_h = 0.15
    ratio_w = 0.2
    im1_warp, im2_warp = warping(im1, im2, im1_points, im2_points, ratio_h, ratio_w)


    # Set the parameters to blend the images into a mosaic.
    kernel_size = (25, 25)
    sigma = 15

    mask = create_alpha_mask(im2_warp, im1_warp)
    mask_gaussian_stack = build_gaussian_stack(mask, kernel_size, sigma, 2)  
    gaussian_stack1 = build_gaussian_stack(im1_warp, kernel_size, sigma, 2)
    laplacian_stack1 = build_laplacian_stack(gaussian_stack1)
    gaussian_stack2 = build_gaussian_stack(im2_warp, kernel_size, sigma, 2)
    laplacian_stack2 = build_laplacian_stack(gaussian_stack2)
    blend_img = blend(laplacian_stack1, laplacian_stack2, mask_gaussian_stack, im1_warp.shape)
    save_normalized_image(blend_img, 'manual_model')


    ''' 
        Manually create mosaic for decoration.
    '''

    imname1 = './media/decoration_left.jpg'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = './media/decoration_right.jpg'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im1_points = np.loadtxt('./media/pts_decoration_left.txt')
    im2_points = np.loadtxt('./media/pts_decoration_right.txt')

    # Set the ratio of the borders.
    ratio_h = 0.15
    ratio_w = 0.4
    im1_warp, im2_warp = warping(im1, im2, im1_points, im2_points, ratio_h, ratio_w)

    # Set the parameters to blend the images into a mosaic.
    kernel_size = (49, 49)
    sigma = 40

    mask = create_alpha_mask(im2_warp, im1_warp)
    mask_gaussian_stack = build_gaussian_stack(mask, kernel_size, sigma, 2)  
    gaussian_stack1 = build_gaussian_stack(im1_warp, kernel_size, sigma, 2)
    laplacian_stack1 = build_laplacian_stack(gaussian_stack1)
    gaussian_stack2 = build_gaussian_stack(im2_warp, kernel_size, sigma, 2)
    laplacian_stack2 = build_laplacian_stack(gaussian_stack2)
    blend_img = blend(laplacian_stack1, laplacian_stack2, mask_gaussian_stack, im1_warp.shape)
    save_normalized_image(blend_img, 'manual_decoration')


    ''' 
        Manually create mosaic for forest.
    '''

    imname1 = './media/forest_left.jpg'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = './media/forest_right.jpg'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im1_points = np.loadtxt('./media/pts_forest_left.txt')
    im2_points = np.loadtxt('./media/pts_forest_right.txt')

    # Set the ratio of the borders.
    ratio_h = 0.15
    ratio_w = 0.2
    im1_warp, im2_warp = warping(im1, im2, im1_points, im2_points, ratio_h, ratio_w)


    # Set the parameters to blend the images into a mosaic.
    kernel_size = (25, 25)
    sigma = 15

    mask = create_alpha_mask(im2_warp, im1_warp)
    mask_gaussian_stack = build_gaussian_stack(mask, kernel_size, sigma, 2)  
    gaussian_stack1 = build_gaussian_stack(im1_warp, kernel_size, sigma, 2)
    laplacian_stack1 = build_laplacian_stack(gaussian_stack1)
    gaussian_stack2 = build_gaussian_stack(im2_warp, kernel_size, sigma, 2)
    laplacian_stack2 = build_laplacian_stack(gaussian_stack2)
    blend_img = blend(laplacian_stack1, laplacian_stack2, mask_gaussian_stack, im1_warp.shape)
    save_normalized_image(blend_img, 'manual_forest')
