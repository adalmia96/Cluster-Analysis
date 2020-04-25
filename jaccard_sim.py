
import string
import random
import numpy as np


def init():
    with open('gmm.txt') as fp:
        gmm = fp.readlines()
    with open('gmmfilt.txt') as fp:
        gmmfilt = fp.readlines()

    gmm_list = []

    gmmfilt_list = []


    for line in gmm:
        line = line.split()[1:]
        if len(line) ==0:
            continue
        gmm_list.append(line)

    for line in gmmfilt:
        line = line.split()[1:]
        if len(line) ==0:
            continue
        gmmfilt_list.append(line)
    total = []
    for i, x in enumerate(gmm_list):
        union = set(gmm_list[i]).union(set(gmmfilt_list[i]))
        inter = set(gmm_list[i]).intersection(set(gmmfilt_list[i]))
        total.append(len(inter)/len(union))
    print(np.mean(total))
if __name__ == "__main__":
    init()
