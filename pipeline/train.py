import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from build_model import build_model, build_simpler_model


def build_pipeline(input_data, BATCH_SIZE, seed=None, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA # Avoid warning messages about sharding

    if shuffle:
        dataset = dataset.shuffle(buffer_size=int(len(input_data)), seed=seed)
    else:
        pass

    return dataset.batch(batch_size=BATCH_SIZE).prefetch(tf.data.AUTOTUNE).with_options(options)


def main():

    # Setup mlflow autolog
    mlflow.tensorflow.autolog()
    mlflow.set_experiment('Test1')

    # Set Training Parameters - TODO: Put all parameters into a config file
    BATCH_SIZE = 32
    EPOCHS = 200
    KERNEL_SIZE = (5, 5)
    INITIAL_NUM_OF_FILTERS = 8
    DROPOUT_RATE = 0.5
    LOSS_FUNCTION = 'binary_focal_crossentropy'
    LEARNING_RATE = 1e-3
    PATIENCE = 10
    SEED = 42
    INPUT_SHAPE = (48, 48, 4)
    tf.random.set_seed(SEED)

    # Load Data
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_valid = np.load("x_valid.npy")
    y_valid = np.load("y_valid.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    # Setup Pipelines
    train_dataset = build_pipeline((x_train, y_train), BATCH_SIZE=BATCH_SIZE, seed=SEED)
    valid_dataset = build_pipeline((x_valid, y_valid), BATCH_SIZE=BATCH_SIZE, shuffle=False)
    test_dataset = build_pipeline((x_test, y_test), BATCH_SIZE=BATCH_SIZE, shuffle=False)

    # set training up on multiple gpus
    input_layer = Input(INPUT_SHAPE)
    output_layer = build_simpler_model(input_layer, INITIAL_NUM_OF_FILTERS, KERNEL_SIZE, DROPOUT_RATE)
    model = Model(input_layer, output_layer)

    METRICS = [
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]

    model.compile(
            loss=LOSS_FUNCTION,
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=METRICS,
        )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=PATIENCE,
        monitor="val_auc",
        mode="max",
        restore_best_weights=True,
    )

    mlflow.log_param('Num_of_parameters', model.count_params())

    history = model.fit(train_dataset, validation_data=valid_dataset,
                        epochs=EPOCHS, callbacks=[early_stop], batch_size=BATCH_SIZE, verbose=1)
    
    model.save("model")


if __name__ == "__main__":

    main()