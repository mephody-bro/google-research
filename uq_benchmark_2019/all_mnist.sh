#!/usr/bin/bash

methods=(vanilla svi ll_svi dropout ll_dropout)

for method in "${methods[@]}"
do
  echo "${method}"

  python -m uq_benchmark_2019.mnist.run_experiment --arch=lenet --method=${method} --output_dir='uq_benchmark_2019/mnist/results/'${method} --mnist_path_tmpl='uq_benchmark_2019/mnist/data/%s.tfrecords' --not_mnist_path_tmpl='uq_benchmark_2019/mnist/data/not_mnist_%s.tfrecords'
done

ENSEMPLE_SIZE=${1:-15}

# Ensemble
#for ((i=0;i<$ENSEMBLE_SIZE;i++)); do
#  echo '*************' ${i}
#  python -m uq_benchmark_2019.mnist.run_experiment --arch=lenet --method=vanilla --output_dir='uq_benchmark_2019/mnist/results/ensemble/'${i} --mnist_path_tmpl='uq_benchmark_2019/mnist/data/%s.tfrecords' --not_mnist_path_tmpl='uq_benchmark_2019/mnist/data/not_mnist_%s.tfrecords'
#done


