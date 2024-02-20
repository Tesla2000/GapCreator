import random
import re
from pathlib import Path

from Config import Config
from ml import get_longest_masks
from string_functions.conv_masked_sentence_to_gap import conv_masked_sentence_to_gap
from string_functions.div_to_sentences import div_to_sentences


def main():
    text_file_path = Path('input.txt')
    sentences = div_to_sentences(text_file_path.read_text())
    for raw_sentence in sentences:
        raw_sentence = re.sub(r'\s+', ' ', raw_sentence)
        sentence = raw_sentence.lower().replace(',', ' ').strip('.?!')
        if len(sentence.split()) > Config.max_sentence_length:
            print(sentence)
            continue
        options = get_longest_masks(sentence)
        if not options:
            print(sentence)
            continue
        masked_sentence = random.choice(options)
        print(conv_masked_sentence_to_gap(raw_sentence, masked_sentence))


if __name__ == '__main__':
    main()
