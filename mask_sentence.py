def mask_sentence(sentence: str, mask_length: int):
    divided_sentences = {}
    for start in range(len(sentence.split()) - mask_length + 1):
        divided_sentence = sentence.split()
        for index in range(start, start + mask_length):
            divided_sentence[index] = "[MASK]"
        divided_sentences[tuple(range(start + 1, start + mask_length + 1))] = ' '.join(divided_sentence)
    return divided_sentences
