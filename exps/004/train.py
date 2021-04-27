import sys
import os
import wandb
import tensorflow as tf


def train() -> None:

    config_defaults = {"layers": 128}
    wandb.init(config=config_defaults, magic=True)

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(wandb.config.layers, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        train_images, train_labels, epochs=5, validation_data=(test_images, test_labels)
    )


if __name__ == "__main__":
    if 1 < len(sys.argv):
        sweep_id = sys.argv[1]
    elif "WANDB_SWEEP_ID" in os.environ:
        sweep_id = os.environ["WANDB_SWEEP_ID"]
    else:
        raise EnvironmentError("missing sweep_id.")
    wandb.agent(sweep_id, function=train)