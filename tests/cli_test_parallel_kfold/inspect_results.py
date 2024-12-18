import pickle
import numpy as np
import matplotlib.pyplot as plt

import glob
import os



for batch in [1,2]:
    results_dir = "/project/3022000.05/projects/stijdboe/temp/parallel_processing/batch_1"


    for func in ['fit', 'predict', 'estimate']:
        print(f"Plotting {func} results...")
        results = glob.glob(os.path.join(results_dir, f"*{func}.pkl"))
        for result in results:
            if "Z" in result:
                z = pickle.load(open(result, "rb"))
                n = np.random.randn(z.shape[0], 1)
                sorted_z = np.sort(z, axis=0)
                sorted_n = np.sort(n, axis=0)
                plt.plot(sorted_z, sorted_n, label=f"Z_{func}")
                plt.savefig(f"Z_{func}.png")
                plt.close()
            elif "yhat" in result:
                x_path = "/project/3022000.05/projects/stijdboe/Projects/PCNtoolkit/tests/cli_test_parallel_kfold/temp/X_te_fcon1000.pkl"
                x = pickle.load(open(x_path, "rb")).to_numpy()
                sortindex = np.argsort(x[:,1])
                print(x[sortindex, 1])
                yhat = pickle.load(open(result, "rb")).to_numpy()
                result = result.replace("yhat", "ys2")
                s2 = pickle.load(open(result, "rb")).to_numpy()
                print(x.shape)
                print(yhat.shape)
                print(s2.shape)

                for i in range(yhat.shape[1]):
                    plt.plot(x[sortindex, 1], yhat[sortindex, i], label=f"Yhat_{func}_{i}")
                    plt.plot(x[sortindex, 1], yhat[sortindex, i] - s2[sortindex, i], label=f"Yhat_{func}_{i} - s2")
                    plt.plot(x[sortindex, 1], yhat[sortindex, i] + s2[sortindex, i], label=f"Yhat_{func}_{i} + s2")
                    plt.savefig(f"Yhat_{func}_ft{i}_batch{batch}.png")
                    plt.close()
            elif "S2" in result:
                s2 = pickle.load(open(result, "rb"))
                print(f"{s2=}")
            elif "EXPV" in result:
                expv = pickle.load(open(result, "rb"))
                print(f"{expv=}")
            elif "MSLL" in result:
                msll = pickle.load(open(result, "rb"))
                print(f"{msll=}")
            elif "SMSE" in result:
                smse = pickle.load(open(result, "rb"))
                print(f"{smse=}")

