import os
import random
import numpy as np

import torch

try:
    import tensorflow as tf

    USE_TENSORFLOW = True

except ModuleNotFoundError:
    USE_TENSORFLOW = False

RANDOM_SEED = 17


def set_all_seeds(seed=RANDOM_SEED):
    # python's seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch's seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if USE_TENSORFLOW:
        # tensorflow's seed
        tf.random.set_seed(seed)


if __name__ == "__main__":
    set_all_seeds(seed=86)
