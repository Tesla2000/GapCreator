import re

from Config import Config


def conv_masked_sentence_to_gap(raw_sentence: str, masked_sentence: list[str]) -> str:
    raw_sentence = raw_sentence.split()
    return re.sub(r'_\s+_', '__', ' '.join(word if mask != Config.mask else 8*'_' for word, mask in zip(raw_sentence, masked_sentence)))
