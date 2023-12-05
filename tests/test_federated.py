from keras.datasets import fashion_mnist
from client import FederatedClient
from server import FederatedServer
from util import build_and_compile_simple_model

# load dataset
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()


def test_client():
    model = build_and_compile_simple_model()
    client_data = train_X[:1000], train_y[:1000]
    client = FederatedClient("test", model, client_data)
    size, weights = client.train(model.get_weights(), epochs=2)
    assert size == 1000
    assert len(weights) == 6

    size, _ = client.train(weights, epochs=2)
    assert size == 1000


def test_server():
    model1 = build_and_compile_simple_model()
    client1_data = train_X[:1000], train_y[:1000]
    client1 = FederatedClient("client1", model1, client1_data)
    model2 = build_and_compile_simple_model()
    client2_data = train_X[1000:2000], train_y[1000:2000]
    client2 = FederatedClient("client2", model2, client2_data)

    server = FederatedServer(build_and_compile_simple_model(), [client1, client2])
    local_weights = server.training_step()
    assert len(local_weights) == 2



