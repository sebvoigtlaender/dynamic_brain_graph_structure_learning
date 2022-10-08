import argparse
import torch as pt

def get_cfg():
    parser = argparse.ArgumentParser(description='')
    cfg = parser.parse_args(args=[])
    # cfg = parser.parse_args()
    return cfg