
import numpy as np
import distance
import nltk


def exact_match_score(references, hypotheses):
    """Computes exact match scores.
    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)
    Returns:
        exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.
    Args:
        references: list of list (one hypothesis)
        hypotheses: list of list (one hypothesis)
    Returns:
        BLEU-4 score: (float)
    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,
                                                   weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def edit_similarity(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += len(ref)

    return 1. - d_leven / len_tot


def compute_scores(refs, hyps):
    scores = {
        "BLEU-4": bleu_score(refs, hyps) * 100,
        "EM": exact_match_score(refs, hyps) * 100,
        "WER": (1 - edit_similarity(refs, hyps)) * 100,
    }
    return scores
