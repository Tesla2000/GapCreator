import os
import random
import sys
from pathlib import Path

from Config import Config
from ml import get_longest_masks
from conv_masked_sentence_to_gap import conv_masked_sentence_to_gap
from div_to_sentences import div_to_sentences


def main():
    text_file_path = Path('input.txt')
    sentences = div_to_sentences(text_file_path.read_text())
    for sentence in sentences:
        if len(sentence.split()) > Config.maks_sentence_length:
            print(sentence, end='. ')
            continue
        masked_sentence = random.choice(get_longest_masks(sentence))
        print(conv_masked_sentence_to_gap(masked_sentence), end='. ')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(os.getcwd()).parent))
    main()
