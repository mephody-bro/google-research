import numpy as np
import scipy
softmax = scipy.special.softmax


# implement BALD approach for dropout confidence
def bald_confidence(stats):
    probs = stats['probs']
    probs_entropy = - np.sum(probs * np.log(probs + 1e-10), axis=-1)

    samples = softmax(stats['logits_samples'], axis=-1)
    samples_entropy = - np.mean(np.sum(samples * np.log(samples + 1e-10), axis=-1), axis=1)

    confidence = -probs_entropy + samples_entropy + 1
    confidence[confidence < 0] = 0

    return confidence

