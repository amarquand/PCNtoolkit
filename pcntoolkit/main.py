import os
from normative_model.norm_conf import NormConf


def main():
    log_dir = "/home/stijn/temp/pcntoolkit/log"
    save_dir = "/home/stijn/temp/pcntoolkit/save"
    conf = NormConf(0, 0, log_dir=log_dir, save_dir=save_dir)
    conf.save_as_json("/home/stijn/temp/pcntoolkit/conf.json")
    del conf
    newconf = NormConf.load_from_json("/home/stijn/temp/pcntoolkit/conf.json")
    print(newconf)

if __name__ == "__main__":
    main()