import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

__version__ = "0.0.1"


n_rows, n_cols = 28, 28
n_classes = 10


def load_data() -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """Load MNIST data, normalize it and convert labels to categorical."""
    # import train and test data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # reshape data
    X_train = X_train.reshape(X_train.shape[0], n_rows, n_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], n_rows, n_cols, 1)

    # set datatype to float and normalize
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Onehot encoding of the classes
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    return (X_train, y_train), (X_test, y_test)


def plot_history(history) -> None:
    """Plot accuracy and validation accuracy."""
    sns.set()
    fig = plt.figure(figsize=(10, 3))

    fig.add_subplot(121)
    sns.lineplot(data=np.array(history.history["loss"]), label="loss")
    sns.lineplot(data=np.array(history.history["val_loss"]), label="val_loss")
    loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]
    plt.title(f"Model loss: {loss:.3f} / {val_loss:.3f}")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    fig.add_subplot(122)
    sns.lineplot(data=np.array(history.history["accuracy"]), label="accuracy")
    sns.lineplot(data=np.array(history.history["val_accuracy"]), label="val_accuracy")
    accuracy = history.history["accuracy"][-1]
    val_accuracy = history.history["val_accuracy"][-1]
    plt.title(f"Model accuracy: {accuracy:.3f} / {val_accuracy:.3f}")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()


def load_handwritten(fn: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load handwritten data."""
    # import
    with open(fn, "rb") as f:
        X_hw, y_hw = pickle.load(f)

    # reshape
    X_hw = X_hw.reshape(X_hw.shape[0], n_rows, n_cols, 1)

    # normalize
    X_hw = 1.0 - (X_hw.astype("float32") / 255.0)

    # categorize
    y_hw = tf.keras.utils.to_categorical(y_hw, n_classes)

    return X_hw, y_hw
