#!/bin/bash

python retrieval/experiments.py --match any --distance_function hamming --hash_method trivial,none --hash_length 64,256
