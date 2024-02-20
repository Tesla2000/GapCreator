import re


def div_to_sentences(raw_text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', raw_text.lower().replace(',', ' '))
    start = 0
    sentences = []
    for item in re.finditer(r'[\.\?!]', text):
        sentence = text[start:item.regs[0][0]].strip()
        start = item.regs[0][1]
        sentences.append(sentence)
    return sentences
