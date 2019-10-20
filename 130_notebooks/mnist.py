def get_mnist():
    return tf.keras.datasets.mnist.load_data()

#%%

plt.imshow(X_train[1], cmap="gray")

#%%

# reshape data
X_train = X_train.reshape(X_train.shape[0], n_rows, n_cols, 1)
X_test = X_test.reshape(X_test.shape[0], n_rows, n_cols, 1)
input_shape = (n_rows, n_cols, 1)

# set datatype to float and normalize
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Onehot encoding of the classes
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)