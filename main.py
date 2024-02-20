from Config import Config
from ml import get_longest_masks
from conv_masked_sentence_to_gap import conv_masked_sentence_to_gap
from div_to_sentences import div_to_sentences


def main():
    sentences = div_to_sentences(
        "When one group of people dominates a field of study, whether it's an age group, a cultural group or a gender, there is danger of creating a narrow view of the subject. Including more women in male-dominated areas will bring in fresh points of view, new talent and creativity. It can also help increase women's social and financial position in some countries. ")
    for sentence in sentences:
        if len(sentence.split()) > Config.maks_sentence_length:
            print("Skipping: ", sentence)
            continue
        for masked_sentence in get_longest_masks(sentence):
            print(conv_masked_sentence_to_gap(masked_sentence))


if __name__ == '__main__':
    main()
