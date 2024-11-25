import os
import pandas as pd
from utils import show_pattern, shuffle_pattern, pattern_diff
from hopfield import Hopfield
import matplotlib.pyplot as plt

folder_path = "data"
files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]
patterns = pd.read_csv(files[0], header=None).values
plt.subplot(1, 3, 1)
show_pattern(patterns[1], (9, 14), False)
noisy_pattern = shuffle_pattern(patterns[1], 0.1)
plt.subplot(1, 3, 2)
show_pattern(noisy_pattern, (9, 14), False)

hebb_net = Hopfield(len(patterns[0]))
hebb_net.train_oja(patterns)

recalled_pattern = hebb_net.update(noisy_pattern, num_iterations=100)
plt.subplot(1, 3, 3)
show_pattern(recalled_pattern, (9, 14), False)

# plt.show()
print("Energy:", hebb_net.energy(recalled_pattern))
print("Diff:", pattern_diff(recalled_pattern, patterns[1]))
