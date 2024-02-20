import random
from pathlib import Path

from Config import Config
from ml import get_longest_masks
from string_functions.conv_masked_sentence_to_gap import conv_masked_sentence_to_gap
from string_functions.div_to_sentences import div_to_sentences


def main():
    text_file_path = Path('input.txt')
    sentences = div_to_sentences(text_file_path.read_text())
    for sentence in sentences:
        if len(sentence.split()) > Config.maks_sentence_length:
            print(sentence, end='. ')
            continue
        options = get_longest_masks(sentence)
        if not options:
            print(sentence, end='. ')
            continue
        masked_sentence = random.choice(options)
        print(conv_masked_sentence_to_gap(masked_sentence), end='. ')


if __name__ == '__main__':
    main()
