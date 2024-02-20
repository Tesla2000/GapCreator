from Config import Config


def mask_sentence(sentence: str, base_sentences: list[str] = None):
    if base_sentences is None:
        masks = tuple((i,) for i in range(len(sentence.split())))
    else:
        masks = tuple(list(i for i in range(len(sentence.split())) if sentence.split()[i] == Config.mask) for sentence in
                      base_sentences)
        masks = tuple(item + [item[-1] + 1] for item in masks)
    divided_sentences = {}
    for mask in masks:
        divided_sentence = sentence.split()
        if len(divided_sentence) == mask[-1]:
            continue
        for index in mask:
            divided_sentence[index] = Config.mask
        divided_sentences[tuple(_ + 1 for _ in mask)] = ' '.join(divided_sentence)
    return divided_sentences
