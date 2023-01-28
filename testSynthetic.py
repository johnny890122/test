#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm


def main():
    dqn = FINDER()
    data_test_path = '../data/synthetic/'
    data_test_name = ['test']
    model_file = './models/nrange_30_50_iter_78000.ckpt'

    file_path = '../results/FINDER_ND/synthetic'

    if not os.path.exists('../results/FINDER_ND'):
        os.mkdir('../results/FINDER_ND')
    if not os.path.exists('../results/FINDER_ND/synthetic'):
        os.mkdir('../results/FINDER_ND/synthetic')
    with open('%s/result.txt'%file_path, 'w') as fout:
        for i in tqdm(range(len(data_test_name))):
            data_test = data_test_path + data_test_name[i]
            val, sol = dqn.Evaluate(data_test, model_file)
            print(val, sol)
            fout.flush()


if __name__=="__main__":
    main()
