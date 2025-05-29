import numpy as np
import skimage as sk
import skimage.io as skio


# Default max displacement
displacement_max = 15

# Default shrinking ratio by each layer
ratio = 3

''' Here are all the functions used to process the images. '''
# Use the sub-image to speed up
# The sub-image is the center of the original one
def sub_image(image, ratio=0.4):
    height, width = image.shape
    return image[int(height * ((1 - ratio) / 2)) : int(height * (1 - (1 - ratio) / 2)),\
                 int(width * ((1 - ratio) / 2)) : int(width * (1 - (1 - ratio) / 2))]

# Automatic cropping
# Remove the black and white borders
def auto_crop(image, black=0.4, white=0.05):
    h, w = image.shape[0], image.shape[1]
    image = image[int(h * 0.05) : int(h * 0.95), int(w * 0.05) : int(w * 0.95)]
    if len(image.shape) == 3:
        mask = np.all(image > black, axis=2) & np.all(image < 1 - white, axis=2)
    else:
        mask = (image > black) & (image < 1 - white)

    top = np.argmax(np.any(mask, axis=1))
    bottom = mask.shape[0] - np.argmax(np.any(mask[::-1], axis=1))
    left = np.argmax(np.any(mask, axis=0))
    right = mask.shape[1] - np.argmax(np.any(mask[:, ::-1], axis=0))

    return image[top:bottom, left:right]

# Automatic white balance
def apply_white_balance(image):
    mean_b = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_r = np.mean(image[:, :, 2])
    mean_gray = (mean_b + mean_g + mean_r) / 3

    scale_b = mean_gray / mean_b
    scale_g = mean_gray / mean_g
    scale_r = mean_gray / mean_r

    image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 1)
    image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 1)
    image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 1)

    return image

# Generate the image pyramids
def build_pyramid(image, scale):
    crop_ratio = 1 / scale
    pyramids = []
    pyramids.append(image)
    while(True):
        new_pyramid = sk.transform.rescale(pyramids[-1], crop_ratio)
        if min(new_pyramid.shape) < displacement_max * 2:
            break
        pyramids.append(new_pyramid)
    return pyramids[::-1]

# Calculate the Normalized Cross-Correlation
def ncc(img1, img2):
    return np.mean(np.multiply(img1 - np.mean(img1), \
                img2 - np.mean(img2))) / (np.std(img1) * np.std(img2))

# Generate all the (x_shift, y_shift) for alignment
def generate_shift_matrix(max_displacement):
    shift_range = np.arange(-max_displacement, max_displacement + 1)
    x_shifts, y_shifts = np.meshgrid(shift_range, shift_range)
    shifts_matrix = np.stack((x_shifts.ravel(), y_shifts.ravel()), axis=-1)
    return shifts_matrix

# Default shift matrix
shift_matrix = generate_shift_matrix(displacement_max)

# Align two channels
def search_align(channel_shift, channel_ref):
    best_score = -np.inf
    best_shift = (0, 0)
    for (x_shift, y_shift) in shift_matrix:
            shifted = np.roll(channel_shift, (x_shift, y_shift), (1, 0))
            score = ncc(shifted, channel_ref)
            if score > best_score:
                best_score = score
                best_shift = (x_shift, y_shift)
    return best_shift

# Apply image pyramid algorithm to two channels
def pyramid_align(channel_shift, channel_ref, scale):
    pyramids_shift = build_pyramid(channel_shift, scale)
    pyramids_ref = build_pyramid(channel_ref, scale)
    best_shift = (0, 0)
    for py_shift, py_ref in zip(pyramids_shift, pyramids_ref):
        shape = py_ref.shape
        if max(shape) > 500:
            py_shift = sub_image(py_shift, 0.6)
            py_ref = sub_image(py_ref, 0.6)
        best_shift = (best_shift[0] * scale, best_shift[1] * scale)
        py_shift = np.roll(py_shift, best_shift, (1, 0))
        shift = search_align(py_shift, py_ref)
        best_shift = (best_shift[0] + shift[0], best_shift[1] + shift[1])
        # print(f'Size {shape}, shift {best_shift}.')
    return best_shift
    
# Process the image alignment and automatic cropping
def align(imname, ratio):
    im = skio.imread(imname)
    im = sk.img_as_float(im)
    
    height = np.floor(im.shape[0] / 3.0).astype(np.int32)
    b = im[: height]
    g = im[height : 2 * height]
    r = im[2 * height : 3 * height]
    B, G, R = sub_image(b), sub_image(g), sub_image(r)

    # Choose green channel as the base
    # print("Start to shifting blue channel:")
    blue_shift = pyramid_align(B, G, ratio)
    print(f"Blue shift: {blue_shift}.\n")

    # print("Start to shifting red channel:")
    red_shift = pyramid_align(R, G, ratio)
    print(f"Red shift: {red_shift}.\n")

    ab = np.roll(b, blue_shift, (1, 0))
    ar = np.roll(r, red_shift, (1, 0))
    im_out = auto_crop(np.dstack([ar, g, ab]))

    return im_out

# Process the whole procedure.
def process(imname):
    im = skio.imread(imname)
    im = sk.img_as_float(im)
    print(f'The original image of {imname}:')
    skio.imshow(im)
    skio.show()

    im_aligned = align(imname, ratio)
    fname = f'./aligned/{imname.split(".")[0]}.jpg'
    im_aligned_uint8 = (im_aligned * 255).astype(np.uint8)
    skio.imsave(fname, im_aligned_uint8)
    print(f'The aligned image without border of {imname}:')
    skio.imshow(im_aligned)
    skio.show()

    im_balance = apply_white_balance(im_aligned)
    fname = f'./balance/{imname.split(".")[0]}.jpg'
    im_balance_uint8 = (im_balance * 255).astype(np.uint8)
    skio.imsave(fname, im_balance_uint8)
    print(f'The image of {imname} after automatic white balance:')
    skio.imshow(im_balance)
    skio.show()
