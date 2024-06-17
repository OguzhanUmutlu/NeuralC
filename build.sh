#!/bin/bash

gcc -o nn ./src/*.c ./src/**/*.c -lm -g -fopenmp -mavx -O3