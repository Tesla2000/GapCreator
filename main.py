import random
import re
from pathlib import Path

from Config import Config
from ml import get_longest_masks
from string_functions.conv_masked_sentence_to_gap import conv_masked_sentence_to_gap
from string_functions.div_to_sentences import div_to_sentences


def main():
    text_file_path = Path('input.txt')
    output_file = Path('output.txt').open('w')
    start_index = 0
    for paragraph in re.finditer(r'<p>([^<]+)</p>', text_file_path.read_text()):
        output_file.write(paragraph.string[start_index:paragraph.regs[1][0]])
        output_file.write('<p>')
        start_index = paragraph.regs[1][1]
        sentences = div_to_sentences(paragraph.string[paragraph.regs[1][0]:start_index])
        for raw_sentence in sentences:
            raw_sentence = re.sub(r'\s+', ' ', raw_sentence)
            write_with_last_sign = lambda sentence: output_file.write(sentence + (raw_sentence[-1] if not sentence.endswith(raw_sentence[-1]) else '') + ' ')
            sentence = raw_sentence.lower().replace(',', ' ').strip('.?!')
            if len(sentence.split()) > Config.max_sentence_length:
                write_with_last_sign(sentence)
                continue
            options = get_longest_masks(sentence)
            if not options:
                write_with_last_sign(sentence)
                continue
            masked_sentence = random.choice(options)
            write_with_last_sign(conv_masked_sentence_to_gap(raw_sentence, masked_sentence))
        output_file.write('</p>')
    output_file.write(paragraph.string[start_index:])
    output_file.close()


if __name__ == '__main__':
    main()
