import dill
import sys  

def load_and_execute(*args):
    with open(args[0], "rb") as executable_path:
        fn = dill.load(executable_path)
    with open(args[1], "rb") as data_path:  
        data = dill.load(data_path)
    fn(data)

if __name__ == "__main__":
    load_and_execute(sys.argv[1:])