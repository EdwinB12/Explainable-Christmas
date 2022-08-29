import numpy as np
import shap
import tensorflow as tf


def main():

    tf.compat.v1.disable_v2_behavior()
    shap.explainers._deep.deep_tf.op_handlers[
        "Conv2DBackpropInput"
    ] = shap.explainers._deep.deep_tf.passthrough

    model = tf.keras.models.load_model("model")

    flattened_model = tf.keras.models.Sequential()
    flattened_model.add(model)
    flattened_model.add(tf.keras.layers.Flatten())
    flattened_model.compile(loss="binary_focal_crossentropy")

    x_train = np.load("x_train.npy")
    x_test = np.load("x_test.npy")

    np.random.seed(42) # always return the same background samples
    background_indexes = np.random.choice(range(x_train.shape[0]), size=100, replace=False)

    background_samples = x_train[
        background_indexes.tolist(), ...
    ]

    explanation_model = shap.DeepExplainer(flattened_model, background_samples)
    expected_values = np.reshape(explanation_model.expected_value, (48, 48))
    np.save("expected_values.npy", expected_values)

    chosen_indexes = [517, 127, 124, 313, 259]
    shap_values = explanation_model.shap_values(x_test[chosen_indexes, ...])
    shap_values = np.array(shap_values)

    np.save("shap_values.npy", shap_values)


if __name__ == "__main__":
    main()