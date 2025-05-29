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

def label_tool(name1, name2):
    imname1 = f'./media/{name1}.jpg'
    im1 = skio.imread(imname1)
    im1 = sk.img_as_float(im1)

    imname2 = f'./media/{name2}.jpg'
    im2 = skio.imread(imname2)
    im2 = sk.img_as_float(im2)

    im1Points, im2Points = select_points(im1, im2)

    np.savetxt(f'./media/points_{name1}.txt', im1Points)
    np.savetxt(f'./media/points_{name2}.txt', im2Points)


if __name__ == "__main__":
    label_tool('george_small', 'DerekPicture')
