from uq_benchmark_2019.array_utils import load_npz
import os


def generate_ensemble_stats(ensemble_dir, stat_name, ensemble_size):
    probs = None
    for i in range(ensemble_size):
        stats = load_npz(os.path.join(ensemble_dir, str(i), stat_name))
        if probs is None:
            probs = stats['probs']
        else:
            probs += stats['probs']
        print(sum(probs[0]))
    probs /= ensemble_size
    return {'probs': probs, 'labels': stats['labels']}