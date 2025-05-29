import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import skimage as sk

def select_points(im1, im2):
    im1Points, im2Points = [], []

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Please select corresponding points on the two images below:')
    axes[0].imshow(im1)
    axes[0].set_title('Image 1')
    axes[1].imshow(im2)
    axes[1].set_title('Image 2')
    
    def on_click(event, image_index):
        if event.inaxes == axes[image_index]:
            if image_index == 0:
                im1Points.append((event.xdata, event.ydata))
                axes[0].plot(event.xdata, event.ydata, 'ro', markersize=2)
                axes[0].text(event.xdata, event.ydata, f'{len(im1Points)}', color='white', fontsize=10)
                plt.draw()
            elif image_index == 1:
                im2Points.append((event.xdata, event.ydata))
                axes[1].plot(event.xdata, event.ydata, 'bo', markersize=2)
                axes[1].text(event.xdata, event.ydata, f'{len(im2Points)}', color='white', fontsize=10)
                plt.draw()

    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, 0))
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, 1))

    plt.show()
    
    return np.array(im1Points), np.array(im2Points)

def label_tool(name1, name2, save_name1, save_name2):
    imname1 = f'./media/{name1}.jpg'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = f'./media/{name2}.jpg'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im1Points, im2Points = select_points(im1, im2)

    np.savetxt(f'./media/{save_name1}.txt', im1Points)
    np.savetxt(f'./media/{save_name2}.txt', im2Points)


if __name__ == "__main__":
    label_tool('building_center', 'building_right', "pts_building_center_cr", "pts_building_right_cr")
    label_tool('building_left', 'building_center', "pts_building_left_lc", "pts_building_center_lc")
    label_tool('street_left', 'street_center', 'pts_street_left_lc', 'pts_street_center_lc')
    label_tool('street_center', 'street_right', 'pts_street_center_cr', 'pts_street_right_cr')
    label_tool('model_left', 'model_center', 'pts_model_left_lc', 'pts_model_center_lc')
    label_tool('model_center', 'model_right', 'pts_model_center_cr', 'pts_model_right_cr')
    label_tool('decoration_left', 'decoration_right', 'pts_decoration_left', 'pts_decoration_right')
    label_tool('forest_left', 'forest_right', 'pts_forest_left', 'pts_forest_right')
    label_tool('phone_lean', 'phone', 'pts_phone_lean', 'pts_phone')
    label_tool('puma_lean', 'puma', 'pts_puma_lean', 'pts_puma')
