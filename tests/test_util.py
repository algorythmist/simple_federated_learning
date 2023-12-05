import numpy as np

from util import shuffle_together, split_data, create_clients


def test_shuffle_together():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 6, 8, 10])
    c = np.array([3, 6, 9, 12, 15])
    a, b, c = shuffle_together(a, b, c)
    for i in range(len(a)):
        assert a[i] == b[i] // 2
        assert a[i] == c[i] // 3


def test_split_data():
    X = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    shards = split_data(X, y, [0.5, 0.2, 0.3])
    assert len(shards) == 3
    assert sum(len(shard[0]) for shard in shards) == 11
    assert len(shards[0][0]) == 5
    assert len(shards[1][0]) == 2
    assert len(shards[2][0]) == 4


def test_create_clients():
    shards = [([1, 2, 3], [4, 5, 6]), ([7, 8, 9], [10, 11, 12])]
    clients = create_clients(shards, lambda: None)
    assert len(clients) == 2
    assert clients[0].train_X == [1, 2, 3]
    assert clients[0].train_y == [4, 5, 6]
    assert clients[1].train_X == [7, 8, 9]
    assert clients[1].train_y == [10, 11, 12]