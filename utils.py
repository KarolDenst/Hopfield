import numpy as np
import matplotlib.pyplot as plt


def shuffle_pattern(pattern, prob):
    noise = np.random.rand(pattern.size) < prob
    noisy_pattern = np.where(noise, -1 * pattern, pattern)

    return noisy_pattern


def show_pattern(pattern, size, show=True):
    pattern = pattern.copy()
    pattern.shape = size
    plt.imshow(pattern, cmap="gray")
    if show:
        plt.show()


def pattern_diff(pattern1, pattern2):
    return np.sum(pattern1 != pattern2)
