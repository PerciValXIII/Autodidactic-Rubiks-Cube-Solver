import pickle

from performance.effectiveness import *

if __name__ == '__main__':
    # np.random.seed(0)
    with open('../nets/trained_net500.pkl', 'rb') as input:
        net = pickle.load(input)

    ncubes, time_per_cube, scramble_range = 40, 720, 30

    # tree_sizes, times, lengths = measure_effectiveness(net, scramble_range, ncubes, time_per_cube)

    with open(f'./results/trees_{ncubes}_{time_per_cube}_{scramble_range}', 'rb') as out:
        tree_sizes = pickle.load(out)
    with open(f'./results/times_{ncubes}_{time_per_cube}_{scramble_range}', 'rb') as out:
        times = pickle.load(out)
    with open(f'./results/lengths_{ncubes}_{time_per_cube}_{scramble_range}', 'rb') as out:
        lengths = pickle.load(out)

    plot_stats(ncubes, scramble_range, times, tree_sizes, lengths)
