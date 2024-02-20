import re


def div_to_sentences(raw_text: str) -> list[str]:
    start = 0
    sentences = []
    for item in re.finditer(r'[\.\?!]', raw_text):
        sentence = raw_text[start:item.regs[0][1]].strip()
        start = item.regs[0][1]
        sentences.append(sentence)
    return sentences
