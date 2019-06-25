
import numpy as np
import distance
import nltk


def exact_match_score(refs, hyps):
    """Computes exact match scores.
    Args:
        refs: list of list of tokens (one ref)
        hyps: list of list of tokens (one hypothesis)
    Returns:
        exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hyp in zip(refs, hyps):
        if np.array_equal(ref, hyp):
            exact_match += 1

    return exact_match / float(max(len(hyps), 1))


def bleu_score(refs, hyps):
    """Computes bleu score.
    Args:
        refs: list of list (one hypothesis)
        hyps: list of list (one hypothesis)
    Returns:
        BLEU-4 score: (float)
    """
    refs = [[ref] for ref in refs]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(refs, hyps,
                                                   weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def edit_similarity(refs, hyps):
    """Computes Levenshtein distance between two sequences.
    Args:
        refs: list of list of token (one hypothesis)
        hyps: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hyp in zip(refs, hyps):
        d_leven += distance.levenshtein(ref, hyp)
        len_tot += len(ref)

    return 1. - d_leven / len_tot


def compute_scores(refs, hyps):
    assert len(refs) == len(hyps)
    scores = {
        "BLEU-4": bleu_score(refs, hyps) * 100,
        "EM": exact_match_score(refs, hyps) * 100,
        "WER": (1 - edit_similarity(refs, hyps)) * 100,
    }
    return scores
