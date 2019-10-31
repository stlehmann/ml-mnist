"""Train dense model with 10x100 hidden units.

## Hyperparameters

* hidden layers: 10
* units per layer: 100
* epochs: 50

## Result metrics

* accuracy: 0.9975
* val_accuracy: 0.9794

"""
# %%
import os
from typing import Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pyprojroot
import seaborn as sns
import tensorflow as tf

sns.set()

# random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

results_p = pyprojroot.here("results/" + os.path.basename(__file__).split(".")[0], [".here"])

# %%
# load mnist dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.astype(np.float32) / 255.0)
X_test= (X_test.astype(np.float32) / 255.0)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype="uint8")
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype="uint8")


# %%
def create_dense_model(
        input_shape: Tuple[int, int] = (64, 64),
        n_hidden: int = 1,
        n_units: int = 7,
        activation: Union[str, Callable] = "relu",
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
        loss: Union[str, tf.keras.losses.Loss] = "binary_crossentropy",
        metrics=None,
) -> tf.keras.Model:
    """Create a dense model.

    :param input_shape: input shape
    :param n_hidden: number of hidden layers
    :param n_units: units per hidden layer
    :param activation: activation function
    :param optimizer: optimizer
    :param loss: loss function
    :param metrics: additional metrics
    :return: model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_units, activation=activation))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

# %%

model = create_dense_model(
    (28, 28),
    n_hidden=10,
    n_units=100,
    activation="relu",
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
)

# %%

sns.lineplot(data=np.array(history.history["accuracy"]), label="accuracy")
sns.lineplot(data=np.array(history.history["val_accuracy"]), label="val_accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()

os.makedirs(results_p)
plt.savefig(results_p / "accuracy.png", dpi=300)
plt.show()
