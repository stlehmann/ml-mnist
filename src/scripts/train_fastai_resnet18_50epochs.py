# %%
import os

import pyprojroot
from fastai.datasets import  URLs, untar_data

from fastai.vision import ImageDataBunch, cnn_learner, accuracy, models

results_p = pyprojroot.here("results/" + str(os.path.basename(__file__).split(".")[0]), [".here"])
os.makedirs(results_p, exist_ok=True)

# %%

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)

# %%

learner = cnn_learner(data, models.resnet18, metrics=[accuracy])

# %%
learner.fit(epochs=10, lr=1e-3)

# %%

learner.save(results_p / "weights")