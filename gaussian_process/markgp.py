import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import gpflow
from typing import Dict, Optional, Tuple
from gpflow.utilities import to_default_float, set_trainable


original_dataset, info = tfds.load(name="mnist", split=tfds.Split.TRAIN, with_info=True)
total_num_data = info.splits["train"].num_examples
image_shape = info.features["image"].shape
image_size = tf.reduce_prod(image_shape)
batch_size = 200


def map_fn(input_slice: Dict[str, tf.Tensor]):
    updated = input_slice
    image = to_default_float(updated["image"]) / 255.0
    label = to_default_float(updated["label"])
    return tf.reshape(image, [-1, image_size]), label


autotune = tf.data.experimental.AUTOTUNE
dataset = (
    original_dataset
    .batch(batch_size, drop_remainder=False)
    .map(map_fn, num_parallel_calls=autotune)
    .prefetch(autotune)
    .repeat()
)

num_mnist_classes = 10
num_inducing_points = 750  # Can also achieve this with 750
images_subset, _ = next(iter(dataset))
images_subset = tf.reshape(images_subset, [-1, image_size])

kernel = gpflow.kernels.SquaredExponential()
likelihood = gpflow.likelihoods.MultiClass(num_mnist_classes)
Z = images_subset.numpy()[:num_inducing_points, :]

model = gpflow.models.SVGP(
    kernel,
    likelihood,
    inducing_variable=Z,
    num_data=total_num_data,
    num_latent_gps=num_mnist_classes,
    whiten=False,
    q_diag=True
)


original_test_dataset, info = tfds.load(name="mnist", split=tfds.Split.TEST, with_info=True)
test_dataset = (
    original_test_dataset
    .batch(10_000, drop_remainder=True)
    .map(map_fn, num_parallel_calls=autotune)
    .prefetch(autotune)
)
test_images, test_labels = next(iter(test_dataset))


data_iterator = iter(dataset)

training_loss = model.training_loss_closure(data_iterator)

logf = []
optimizer = tf.optimizers.Adam(learning_rate=1e-4)


@tf.function
def optimization_step():
    optimizer.minimize(training_loss, model.trainable_variables)


f = open("training_results.txt", "w")
f.write("elbo accuracy variance lengthscale\n")
print("Starting optimisation...")
for step in range(1000000):
    optimization_step()
    if step % 1000 == 0:
        elbo = -training_loss().numpy()
        logf.append(elbo)

        m, v = model.predict_y(test_images)
        preds = np.argmax(m, 1).reshape(test_labels.numpy().shape)
        correct = preds == test_labels.numpy().astype(int)
        acc = np.average(correct.astype(float)) * 100.0

        print("ELBO: {:.4e}, Accuracy is {:.4f}%".format(elbo, acc))
        f.write("{:.4e} {:.4f} {:.4f} {:.4f}\n".format(
            elbo,
            acc,
            model.kernel.variance.numpy(),
            model.kernel.lengthscales.numpy())
        )
f.close()
