import os
import sys

import cloudpickle as pickle


def main():
    print("Submitted batch job ", os.getpid())

    callable_file = sys.argv[1]
    data_file = sys.argv[2]
    
    # Load the pickled function and data
    with open(callable_file, 'rb') as f:
        fn = pickle.load(f)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        fn(data[0], data[1])
    else:
        fn(data)


if __name__ == '__main__':
    main() 