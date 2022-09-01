import numpy as np
import shap
import tensorflow as tf
import time
import mlflow


def main():

    # Set some required parameters and settings
    tf.compat.v1.disable_v2_behavior()
    shap.explainers._deep.deep_tf.op_handlers[
        "Conv2DBackpropInput"
    ] = shap.explainers._deep.deep_tf.passthrough

    # Load model
    model = tf.keras.models.load_model("outputs/model")

    # Model must be flattened
    flattened_model = tf.keras.models.Sequential()
    flattened_model.add(model)
    flattened_model.add(tf.keras.layers.Flatten())
    flattened_model.compile(loss="binary_focal_crossentropy")

    # Load training and test inputs
    x_train = np.load("outputs/x_train.npy")[:, ::2, ::2, :]
    x_test = np.load("outputs/x_test.npy")[:, ::2, ::2, :]

    # Set seed so code is reproducible
    np.random.seed(42)

    # Random set of 100 samples go into calculating the expected values
    background_indexes = np.random.choice(range(x_train.shape[0]), size=100, replace=False)
    background_samples = x_train[
        background_indexes.tolist(), ...
    ]

    # Create expected values and save
    start = time.time()
    explanation_model = shap.DeepExplainer(flattened_model, background_samples)
    expected_values = np.reshape(explanation_model.expected_value, (24, 24))
    print(f'Expected values took {(time.time() - start)/60} Minutes to create')
    np.save("outputs/expected_values.npy", expected_values)

    # For chosen example from the test dataset, generate shap values. For 10 images at 24x24x4, this took 2 hours on a laptop
    start = time.time()
    chosen_indexes = [127, 527, 259, 401, 210, 90, 610, 600, 24]
    print(f'SHAP values for {len(chosen_indexes)} examples are being created. This may take a while. . ')
    shap_values = explanation_model.shap_values(x_test[chosen_indexes, ...])
    shap_values = np.array(shap_values)
    print(f'Shap Values took {(time.time() - start) / 60} Minutes to create')
    np.save("outputs/shap_values.npy", shap_values)
    np.save('outputs/test_indexes_used_for_SHAP.npy', np.array(chosen_indexes, dtype='int'))

    # Add Shap artifacts
    mlflow.log_artifact('outputs/shap_values.npy', artifact_path="outputs")
    mlflow.log_artifact('outputs/test_indexes_used_for_SHAP.npy', artifact_path="outputs")
    mlflow.log_artifact('outputs/expected_values.npy', artifact_path="outputs")

    # Add Jobs to artifacts
    mlflow.log_artifacts('Jobs', artifact_path="Jobs")


if __name__ == "__main__":
    main()