import numpy as np
import shap
import tensorflow as tf
import pickle
import time
import dill


def main():

    tf.compat.v1.disable_v2_behavior()
    shap.explainers._deep.deep_tf.op_handlers[
        "Conv2DBackpropInput"
    ] = shap.explainers._deep.deep_tf.passthrough

    model = tf.keras.models.load_model("outputs/model")

    flattened_model = tf.keras.models.Sequential()
    flattened_model.add(model)
    flattened_model.add(tf.keras.layers.Flatten())
    flattened_model.compile(loss="binary_focal_crossentropy")

    x_train = np.load("outputs/x_train.npy")[:, ::2, ::2, :]
    x_test = np.load("outputs/x_test.npy")[:, ::2, ::2, :]

    np.random.seed(42) # always return the same background samples
    background_indexes = np.random.choice(range(x_train.shape[0]), size=100, replace=False)

    background_samples = x_train[
        background_indexes.tolist(), ...
    ]

    start = time.time()
    explanation_model = shap.DeepExplainer(flattened_model, background_samples)
    expected_values = np.reshape(explanation_model.expected_value, (24, 24))
    print(f'Expected values took {(time.time() - start)/60} Minutes to create')
    np.save("outputs/expected_values.npy", expected_values)

    start = time.time()
    chosen_indexes = [127, 527, 259, 401, 210, 90, 610, 600, 24]
    print(f'SHAP values for {len(chosen_indexes)} examples are being created. This may take a while. . ')
    shap_values = explanation_model.shap_values(x_test[chosen_indexes, ...])
    shap_values = np.array(shap_values)
    print(f'Shap Values took {(time.time() - start) / 60} Minutes to create')
    np.save("outputs/shap_values.npy", shap_values)


if __name__ == "__main__":
    main()