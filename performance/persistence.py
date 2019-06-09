import pickle

from adi.fullnet import FullNet


def save_net(net: FullNet, filename: str):
    with open(filename, 'wb') as output:
        pickle.dump(net, output)


def load_net(filename: str) -> FullNet:
    with open(filename, 'rb') as input:
        return pickle.load(input)
