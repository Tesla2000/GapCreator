import re


def conv_masked_sentence_to_gap(masked_sentence: str) -> str:
    masked_sentence = masked_sentence.replace('[MASK]', 10*"_")
    return re.sub(r'_\s+_', '__', masked_sentence)
