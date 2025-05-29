import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.io as skio
import skimage as sk

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, (dscale, dscale, 1)) # should be same as passing multichannel=True
    else:
        im2 = sktr.rescale(im2, (1./dscale, 1./dscale, 1))
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    # assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

# Save the normalized image.
def save_normalized_image(image, imname):
    im_normalized = (image - image.min()) / (image.max() - image.min())
    im_uint8 = (im_normalized * 255).astype(np.uint8)
    fname = f'./media/{imname}.jpg'
    skio.imsave(fname, im_uint8)
    return im_uint8


if __name__ == "__main__":
    # Generate the first group of aligned images.
    imname1 = './media/DerekPicture.jpg'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = './media/nutmeg.jpg'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im2_aligned, im1_aligned = align_images(im2, im1)
    save_normalized_image(im1_aligned, 'aligned_low1')
    save_normalized_image(im2_aligned, 'aligned_high1')


    # Generate the second group of aligned images.
    imname1 = './media/Monroe.png'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = './media/Einstein.png'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im2_aligned, im1_aligned = align_images(im2, im1)
    save_normalized_image(im1_aligned, 'aligned_low2')
    save_normalized_image(im2_aligned, 'aligned_high2')


    # Generate the third group of aligned images.
    imname1 = './media/wolf.jpg'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = './media/panda.jpg'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im2_aligned, im1_aligned = align_images(im2, im1)
    save_normalized_image(im1_aligned, 'aligned_low3')
    save_normalized_image(im2_aligned, 'aligned_high3')


    # Generate the fourth group of aligned images.
    imname1 = './media/dog.jpg'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = './media/cat.jpg'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im2_aligned, im1_aligned = align_images(im2, im1)
    save_normalized_image(im1_aligned, 'aligned_low4')
    save_normalized_image(im2_aligned, 'aligned_high4')
