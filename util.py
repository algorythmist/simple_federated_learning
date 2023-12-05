import keras
import numpy as np

from client import FederatedClient


def shuffle_together(*arrays, seed=None):
    """
    Shuffle arrays in unison
    :param arrays: a tuple or list of arrays. They must all have the same length
    :param seed: random seed
    :return: a tuple of the shuffled arrays
    """
    if seed is not None:
        np.random.seed(seed)
    if len(arrays) == 0:
        return []
    p = np.random.permutation(len(arrays[0]))
    return (a[p] for a in arrays)


def split_data(features, labels, shard_ratios):
    """
    Split the data into shards according to the specified ratios of the total size
    :param features: The features (predictors)
    :param labels: The labels
    :param shard_ratios: an array of ratios that sum to 1.0
    :return: a list of tuples (X, y)
    """
    assert np.isclose(sum(shard_ratios), 1.0)
    total_rows = len(labels)
    # determine the size of each shard
    shard_sizes = [int(total_rows * ratio) for ratio in shard_ratios]
    total = sum(shard_sizes)
    if total < total_rows:
        # allocate the remaining points to the last shard
        shard_sizes[-1] += (total_rows - total)
    index = 0
    subsets = []
    for shard_size in shard_sizes:
        subsets.append((features[index: index + shard_size], labels[index: index + shard_size]))
        index += shard_size
    return subsets


def build_simple_model():
    """
    A simple model for classifying 28x28 images
    :return:
    """
    return keras.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Rescaling(scale=1 / 255),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


def build_and_compile_simple_model():
    """
    A simple model for classifying 28x28 images, compiled for training
    :return:
    """
    model = build_simple_model()
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer='sgd',
                  metrics=[keras.metrics.sparse_categorical_accuracy])
    return model


def create_clients(shards, prefix='client', create_model_fn=build_and_compile_simple_model):
    """
    Create one client per shard of data
    :param shards: A shard of training data with labels
    :param prefix: the prefix for the client name
    :param create_model_fn: A function that creates a model
    :return:
    """
    num_clients = len(shards)
    # create a list of client names
    client_names = ['{}_{}'.format(prefix, i + 1) for i in range(num_clients)]
    return [FederatedClient(client_names[i], create_model_fn(), shards[i])
            for i in range(len(client_names))]
