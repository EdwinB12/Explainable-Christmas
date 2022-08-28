import numpy as np
from sklearn import preprocessing, model_selection
import imageio
import os


def load_data_into_arrays(stem):
    images = []
    masks = []

    for root, dirs, files in sorted(os.walk(stem, topdown=False)):
        for name in files:

            filename = os.path.join(root, name)

            if 'mask' in filename:
                masks.append(np.expand_dims(imageio.imread(filename), -1))
            elif 'images' in filename:
                images.append(imageio.imread(filename)[:, :, 0:1])
            else:
                pass

    images = np.asarray(images, dtype='float32')
    masks = np.asarray(masks, dtype='float32')
    return images, masks


def plot_im(im, ax, title):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)


def plot_mask(mask, ax, title):
    ax.imshow(mask)
    ax.set_title(title)


def run():
    """ Segregate and transform the data into training and testing sets."""

    images, masks = load_data_into_arrays('../Data/train')
    print(f'Images have been loaded with shape: {images.shape}')
    print(f'Masks have been loaded with shape: {masks.shape}')

    # Reduce dimensions for speed up of training and shape
    images = images[:, ::2, ::2]
    masks = masks[:, ::2, ::2]
    print(f'Images have been resampled. Images now have shape: {images.shape}')
    print(f'Masks have been resampled. Masks now have shape: {masks.shape}')

    # to recreate the original shape we'll need to store it
    n_samples, img_x, img_y, n_channels = images.shape

    # train-test split
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        images, masks, test_size=0.2
        )

    del images
    del masks

    # train-test split
    x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
        x_train, y_train, test_size=0.2
        )

    print(f'Data is split into train, validation and test datasets with number of samples:'
          f' {len(x_train), len(x_valid), len(x_test)}')

    return x_train, y_train, x_valid, y_valid, x_test, y_test


if __name__ == "__main__":

    run()