from server import FederatedServer
from util import *

if __name__ == '__main__':
    dataset = keras.datasets.fashion_mnist

    (X_train, y_train), (X_test, y_test) = dataset.load_data()

    # shuffle the training data
    shuffle_together(X_train, y_train)

    # split the training data between the server and the clients
    shard_ratios = [0.2, 0.3, 0.1, 0.2, 0.2]
    shards = split_data(X_train, y_train, shard_ratios)
    # assign the first shard to the server for pre-training
    X_server, y_server = shards[0]
    clients = create_clients(shards=shards[1:], create_model_fn=build_and_compile_simple_model)

    server_model = build_and_compile_simple_model()

    server = FederatedServer(server_model, clients, batch_size=10, client_epochs=1)
    server.train(10, evaluate_fn=lambda model: model.evaluate(X_server, y_server))
    loss, accuracy = server_model.evaluate(X_test, y_test)
    print(f" accuracy: {accuracy} | loss: {loss}")
