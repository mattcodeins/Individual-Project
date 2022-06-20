import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import gpflow
from typing import Dict
from gpflow.utilities import to_default_float


original_dataset, info = tfds.load(name="mnist", split=tfds.Split.TRAIN, with_info=True)
total_num_data = info.splits["train"].num_examples
image_shape = info.features["image"].shape
image_size = tf.reduce_prod(image_shape)
batch_size = 200

def map_fn(input_slice: Dict[str, tf.Tensor]):
    updated = input_slice
    image = to_default_float(updated["image"]) / 255.0
    label = tf.one_hot(int(updated["label"]), 10, dtype='int32')
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
num_inducing_points = 750
is1, _ = next(iter(dataset))
is2, _ = next(iter(dataset))
is3, _ = next(iter(dataset))
is4, _ = next(iter(dataset))
images_subset = np.concatenate((is1, is2, is3, is4))

Z = images_subset[:num_inducing_points, :]
print(Z.shape)

kernel = gpflow.kernels.SquaredExponential()

class MySoftmax(gpflow.likelihoods.Softmax):
    def _log_prob(self, F, Y):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=tf.argmax(Y, axis=1))

likelihood = MySoftmax(num_mnist_classes)

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
# lr_sched = tf.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-5,
#     decay_steps=1000,
#     decay_rate=0.8)
boundaries = list(np.arange(1,5)*2e4)
print(boundaries)
values = list(10 ** np.arange(-2,-7,-1).astype(float))
print(values)
lr_sched = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values
)
optimizer = tf.optimizers.Adam(learning_rate=lr_sched)


@tf.function
def optimization_step():
    optimizer.minimize(training_loss, model.trainable_variables)


f = open("training_results.txt", "w")
f.write("elbo accuracy variance lengthscale\n")
print("Starting optimisation...")
for step in range(100_000):
    optimization_step()
    if step % 1000 == 0:
        elbo = -training_loss().numpy()
        logf.append(elbo)

        m, v = model.predict_y(test_images)
        preds = tf.one_hot(np.argmax(m, 1), 10, dtype='int32')
        correct = preds == test_labels.numpy()
        acc = np.average(correct) * 100.0
        lr = optimizer._decayed_lr(tf.float32)
        print("Step: {:.0f}, Learning Rate: {:.2e}, ELBO: {:.4e}, Accuracy: {:.4f}%".format(step, lr, elbo, acc))
        f.write("{:.0f} {:.4e} {:.4f} {:.4f} {:.4f}\n".format(
            step,
            elbo,
            acc,
            model.kernel.variance.numpy(),
            model.kernel.lengthscales.numpy()
            )
        )
f.close()
