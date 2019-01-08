from scipy import sparse
from scipy.stats import uniform
import numpy as np
from datetime import datetime

import random

SIZE = 640 * 480
# SIZE = 15


def create_coo():
    # non_zeros = random.randint(np.floor(np.sqrt(SIZE) - np.log(SIZE)), np.floor(np.sqrt(SIZE) + np.log(SIZE)))
    non_zeros = random.randint(np.floor(SIZE ** 1.2 - np.log(SIZE)), np.floor(SIZE ** 1.2 + np.log(SIZE)))
    print(datetime.now(), non_zeros)
    rows = np.array(random.choices(range(SIZE), k=non_zeros))
    print(datetime.now())
    cols = np.array(random.choices(range(SIZE), k=non_zeros))
    print(datetime.now())
    fill = [np.random.randn() for _ in range(non_zeros)]
    print(datetime.now())
    return sparse.coo_matrix((fill, (rows, cols)), shape=(SIZE, SIZE))


if __name__ == '__main__':
    print(datetime.now())
    print(create_coo().multiply(create_coo()))
    print(datetime.now())
