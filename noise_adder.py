#!/usr/bin/env python3

import argparse
import math

import numpy as np
import Levenshtein as levenshtein

from scipy.stats import poisson
from tqdm import tqdm


def truncated_poisson_pmf(lambda_value, k, n):
    return poisson.pmf(k, lambda_value) / sum(poisson.pmf(i, lambda_value) for i in range(0, n + 1))


def draw_from_truncated_poisson(lambda_value, n):
    truncated_poisson_distribution = [truncated_poisson_pmf(lambda_value, k, n) for k in range(n + 1)]
    return np.random.choice(n + 1, p=truncated_poisson_distribution)


def discrete_simplex(d, e, buckets_distribution=None):
    if buckets_distribution is not None and len(buckets_distribution) != d:
        raise ValueError('{} bucket sizes must be given'.format(d))

    d = d if buckets_distribution is None else sum(buckets_distribution)

    x = np.random.choice(range(1, e + d), size=d - 1, replace=False)
    x = [0] + sorted(x) + [e + d]

    n = [x[i] - x[i - 1] - 1 for i in range(1, d + 1)]

    if buckets_distribution is not None:
        n_from_multiple_buckets = []
        start_index = 0
        for buckets in buckets_distribution:
            end_index = start_index + buckets
            n_from_multiple_buckets.append(sum(n[start_index:end_index]))
            start_index = end_index
        return n_from_multiple_buckets

    return n


def sample_from_unigram_distribution(vocab):
    vocab_sum = sum(frequency for _, frequency in vocab)
    word_prob = [frequency / vocab_sum for _, frequency in vocab]
    return vocab[np.random.choice(len(vocab), p=word_prob)][0]


def sample_word_based_on_acoustic_sim(vocab, word_to_replace):
    distances = [math.exp(-levenshtein.distance(word_to_replace, word_vocab)) for word_vocab in vocab]
    sum_distances = sum(distances)
    word_prob = [distance / sum_distances for distance in distances]
    return vocab[np.random.choice(len(vocab), p=word_prob)]


def sample_word(vocab, unigram_distribution=False, acoustic_distribution=False, word_to_replace=None):
    if unigram_distribution and acoustic_distribution:
        raise NotImplementedError('usage of unigram_distribution and acoustic_distribution together is not implemented')

    if acoustic_distribution and word_to_replace is None:
        raise ValueError('a word to replace must be given if a word should be sampled based on acoustic similarity')

    if unigram_distribution:
        return sample_from_unigram_distribution(vocab)
    elif acoustic_distribution:
        return sample_word_based_on_acoustic_sim(vocab, word_to_replace)

    return vocab[np.random.choice(len(vocab))]


def sample_position_based_on_length(utterance, number_positions, ignore_positions=[]):
    lengths = []
    for i, word in enumerate(utterance):
        if i in ignore_positions:
            lengths.append(0)
        else:
            lengths.append(math.exp(-len(word)))

    sum_lengths = sum(lengths)
    word_prob = [length / sum_lengths for length in lengths]
    return np.random.choice(len(utterance), size=number_positions, replace=False, p=word_prob)


def sample_positions(utterance, number_positions, length_distribution=False, ignore_positions=[]):
    if number_positions == 0:
        return []

    if length_distribution:
        return sample_position_based_on_length(utterance, number_positions, ignore_positions=ignore_positions)

    return np.random.choice(list(set(range(len(utterance))).difference(ignore_positions)),
                            size=number_positions, replace=False)


def induce_noise_utterance(utterance, vocab, args):
    utterance_splitted = utterance.split()
    n = len(utterance_splitted)

    e = draw_from_truncated_poisson(args.tau * n, n)

    n_s, n_i, n_d = discrete_simplex(3, e, args.buckets_distribution)

    substitution_positions = sample_positions(utterance_splitted, n_s, args.length_distribution)

    insertion_positions = np.random.choice(n + 1 - n_d, size=n_i, replace=False)

    deletion_positions = sample_positions(utterance_splitted, n_d, args.length_distribution,
                                          ignore_positions=substitution_positions)

    for substitution_position in substitution_positions:
        substitution_word = sample_word(vocab, args.unigram_distribution, args.acoustic_distribution,
                                        utterance_splitted[substitution_position])
        utterance_splitted[substitution_position] = substitution_word

    for deletion_position in sorted(deletion_positions, reverse=True):
        utterance_splitted.pop(deletion_position)

    for insertion_position in sorted(insertion_positions, reverse=True):
        insertion_word = sample_word(vocab, unigram_distribution=args.unigram_distribution)
        utterance_splitted.insert(insertion_position, insertion_word)

    return ' '.join(utterance_splitted)


def induce_noise(args):
    with open(args.vocab_path, 'r') as vocab_file:
            vocab = []
            for line in vocab_file.read().splitlines():
                vocab_line_splitted = line.split('\t', 1)
                if args.unigram_distribution:
                    vocab.append((vocab_line_splitted[0], int(vocab_line_splitted[1])))
                else:
                    vocab.append(vocab_line_splitted[0])

    with open(args.dataset_path, 'r') as dataset_file:
        number_utterances = 0
        for number_utterances, _ in enumerate(dataset_file, 1):
            pass

    with open(args.dataset_path, 'r') as dataset_file:
        with tqdm(total=number_utterances, unit=' utterances') as progress_bar:
            for _, line in enumerate(dataset_file):
                print(induce_noise_utterance(line, vocab, args))
                progress_bar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Induces noise to given utterances (based on the paper Toward '
                                     'Robust Neural Machine Translation for Noisy Input Sequences by Matthias '
                                     'Sperber, Jan Niehues, and Alex Waibel).')
    parser.add_argument('dataset_path', metavar='DATASET_FILE', type=str,
                        help='file with the utterances (separated by newline) to that noise should be induced')
    parser.add_argument('vocab_path', metavar='VOCAB_FILE', type=str,
                        help='file with the vocabulary (separated by newline), if unigram_distribution is used, ' \
                        'a tab and the number of occurrences must be added after each word)')
    parser.add_argument('-t', '--tau', metavar='TAU_VALUE', type=float, default=0.08,
                        help='tau value (in the range from 0 to 1, the higher the tau value, ' \
                        'the more noise is induced)')
    parser.add_argument('-u', '--unigram_distribution', action='store_true',
                        help='sample replacement and insertion words from a unigram distribution')
    parser.add_argument('-a', '--acoustic_distribution', action='store_true',
                        help='sample replacement and insertion words from a distribution based on acoustic similarity')
    parser.add_argument('-l', '--length_distribution', action='store_true',
                        help='sample substitution and deletion positions from a distribution based on word lengths ' \
                        '(shorter words are chosen more likely)')
    parser.add_argument('-b', '--buckets_distribution', metavar=('SUB_BUCKETS', 'INS_BUCKETS', 'DEL_BUCKETS'),
                        type=int, nargs=3, default=[1, 1, 1], help='buckets distribution to substitution, insertion, ' \
                        'and deletion')
    args = parser.parse_args()

    induce_noise(args)
