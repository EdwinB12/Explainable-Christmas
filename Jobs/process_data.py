import time
from sklearn import preprocessing
import bruges as bg
import numpy as np
from scipy.ndimage import sobel
from skimage.transform import resize
import read_and_split_data
import pickle
import os


def rms(data):
    data = np.asanyarray(data)
    return np.sqrt(np.sum(data**2) / data.size)


def attribute_wrapper(input_data, attribute_func, output_shape=None, **kwargs):
    image = np.reshape(input_data, newshape=(1, input_data.shape[0], input_data.shape[1]))
    attribute = attribute_func(image, **kwargs)
    
    if output_shape is None:
        return np.reshape(attribute, input_data.shape)
    else:
        return np.reshape(attribute, output_shape)

    
def get_attributes_for_single_image(image):
    energy = attribute_wrapper(image, bg.attribute.energy, duration=5, dt=1)
    semb = attribute_wrapper(image, bg.attribute.similarity, duration=5, dt=1, step_out=3, kind='gst')
    sobel_image = sobel(image)

    return np.stack([image, energy, semb, sobel_image], axis=-1)


def get_attributes_for_multiple_images(multiple_image):
    now = time.time()
    attributes = np.zeros(shape=(*multiple_image.shape[:3], 4), dtype=np.float32)
    for index in range(multiple_image.shape[0]):
        attributes[index] = get_attributes_for_single_image(multiple_image[index])

        if index % 100 == 0:
            print(f'{index}th Image Complete in {(time.time() - now)/60} Minutes')

    print(f'Attributes Finished')
    return attributes


def remove_nan_samples_and_resize(attributes, y, output_shape):
    attributes_x_reshaped = []
    y_reshaped = []

    for i in range(attributes.shape[0]):
        if np.isnan(attributes[i]).any():
            pass
        else:
            attributes_x_reshaped.append(resize(attributes[i], (*output_shape, 4), anti_aliasing=True))
            y_reshaped.append(resize(y[i], (*output_shape, 1), anti_aliasing=True))

    return np.asarray(attributes_x_reshaped, dtype=np.float32), np.asarray(y_reshaped, dtype=np.float32)


def main():

    # Read and Split Data
    x_train, y_train, x_valid, y_valid, x_test, y_test = read_and_split_data.run()

    x_train = resize(x_train, output_shape=(len(x_train), 48, 48, 1), anti_aliasing=True)
    y_train = resize(y_train, output_shape=(len(y_train), 48, 48, 1), anti_aliasing=True)
    x_valid = resize(x_valid, output_shape=(len(x_valid), 48, 48, 1), anti_aliasing=True)
    y_valid = resize(y_valid, output_shape=(len(y_valid), 48, 48, 1), anti_aliasing=True)
    x_test = resize(x_test, output_shape=(len(x_test), 48, 48, 1), anti_aliasing=True)
    y_test = resize(y_test, output_shape=(len(y_test), 48, 48, 1), anti_aliasing=True)
    print('Data is Resized')
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    # Generate Attributes for training, validation and test datasets
    x_train = get_attributes_for_multiple_images(x_train[:, :, :, 0])
    x_valid = get_attributes_for_multiple_images(x_valid[:, :, :, 0])
    x_test = get_attributes_for_multiple_images(x_test[:, :, :, 0])
    print('Attributes Built')
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    nan_idx = np.isnan(x_train).any(axis=(1, 2, 3))
    x_train = x_train[~nan_idx]
    y_train = y_train[~nan_idx]

    nan_idx = np.isnan(x_valid).any(axis=(1, 2, 3))
    x_valid = x_valid[~nan_idx]
    y_valid = y_valid[~nan_idx]

    nan_idx = np.isnan(x_test).any(axis=(1, 2, 3))
    x_test = x_test[~nan_idx]
    y_test = y_test[~nan_idx]
    print('Nans Dropped')

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)
    
    _, x_img_x, x_img_y, x_nchannels = x_train.shape

    # Fit transformer to the training set then apply it to test set
    transformer = preprocessing.PowerTransformer()
    x_train = transformer.fit_transform(x_train.reshape(-1, x_nchannels))
    x_train = x_train.reshape(-1, x_img_x, x_img_y, x_nchannels)

    x_valid = transformer.transform(x_valid.reshape(-1, x_nchannels))
    x_valid = x_valid.reshape(-1, x_img_x, x_img_y, x_nchannels)

    x_test = transformer.transform(x_test.reshape(-1, x_nchannels))
    x_test = x_test.reshape(-1, x_img_x, x_img_y, x_nchannels)

    # Change Y labels to 0s and 1s
    y_train = np.where(y_train == 0, 0, 1)
    y_valid = np.where(y_valid == 0, 0, 1)
    y_test = np.where(y_test == 0, 0, 1)

    print(f'Datasets are pre-conditioned')

    if os.path.exists('outputs'):
        pass
    else:
        os.mkdir('outputs')

    np.save('outputs/x_train', x_train)
    np.save('outputs/y_train', y_train)
    np.save('outputs/x_valid', x_valid)
    np.save('outputs/y_valid', y_valid)
    np.save('outputs/x_test', x_test)
    np.save('outputs/y_test', y_test)

    with open('outputs/x_transformer.pkl', 'wb') as file:
        pickle.dump(transformer, file)


if __name__ == "__main__":

    main()