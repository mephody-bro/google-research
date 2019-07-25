#!/usr/bin/bash

METHOD=$1


python -m uq_benchmark_2019.mnist.run_experiment --arch=lenet --method=$METHOD --output_dir='uq_benchmark_2019/mnist/results/$METHOD' --mnist_path_tmpl='uq_benchmark_2019/mnist/data/%s.tfrecords' --not_mnist_path_tmpl='uq_benchmark_2019/mnist/data/not_mnist_%s.tfrecords'
