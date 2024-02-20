import re
from itertools import count

import torch
from transformers import BertTokenizer, BertForMaskedLM

from Config import Config
from string_functions.mask_sentence import mask_sentence

enc = BertTokenizer.from_pretrained('bert-base-uncased')

batch_size = 4

batched_indexed_tokens = [[101, 64] * 64] * batch_size
batched_segment_ids = [[0, 1] * 64] * batch_size
batched_attention_masks = [[1, 1] * 64] * batch_size

tokens_tensor = torch.tensor(batched_indexed_tokens)
segments_tensor = torch.tensor(batched_segment_ids)
attention_masks_tensor = torch.tensor(batched_attention_masks)
mlm_model_ts = BertForMaskedLM.from_pretrained('bert-base-uncased', torchscript=True)
traced_mlm_model = torch.jit.trace(mlm_model_ts, [tokens_tensor, segments_tensor, attention_masks_tensor])
stripped_signs = '.?!,'


def get_longest_masks(sentence: str) -> tuple[str, ...]:
    previous_match = None
    for mask_length in count(1):
        sentence = sentence.strip(stripped_signs).lower()
        masked_sentences = mask_sentence(sentence, previous_match)
        pos_masks = tuple(masked_sentences.keys())

        origin_masked_sentences = tuple(masked_sentences.values())
        masked_sentences = tuple(origin_masked_sentences)

        for position_index in range(mask_length):
            unmasked_tokens = []
            for sentence_start in range(0, len(masked_sentences), Config.versions_calculated_at_once):
                encoded_inputs = enc(
                    masked_sentences[sentence_start:sentence_start + Config.versions_calculated_at_once],
                    return_tensors='pt', padding='max_length', max_length=128)
                outputs = mlm_model_ts(**encoded_inputs)
                most_likely_token_ids = [
                    torch.argmax(
                        outputs[0][i, positions[position_index]]) for i, positions in enumerate(
                        pos_masks[sentence_start:sentence_start + Config.versions_calculated_at_once])
                ]
                unmasked_tokens += list(enc.decode([token]) for token in most_likely_token_ids)
            masked_sentences = tuple(masked_sentences[sentence_index].replace('[MASK]', token.strip(stripped_signs), 1)
                                     for sentence_index, token in enumerate(unmasked_tokens))
        matching_sentences = tuple(origin_masked_sentence for masked_sentence, origin_masked_sentence in
                                   zip(masked_sentences, origin_masked_sentences) if
                                   re.sub(r'\s+', ' ', masked_sentence) == re.sub(r'\s+', ' ', sentence))
        if not matching_sentences:
            return previous_match
        previous_match = matching_sentences
