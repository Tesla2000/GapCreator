from itertools import count

import torch
import torch.nn.functional as F
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
mlm_model_ts.eval()
stripped_signs = '.?!,'


@torch.no_grad
def get_longest_masks(base_sentence: str) -> tuple[str, ...]:
    previous_match = None
    while True:
        base_sentence = base_sentence.strip(stripped_signs).lower()
        masked_sentences = mask_sentence(base_sentence, previous_match)
        pos_masks = tuple(masked_sentences.keys())
        masked_sentences = list(masked_sentences.values())
        if pos_masks:
            if len(pos_masks[0]) > Config.max_gap_length:
                return previous_match
        original_sentences = tuple(masked_sentences)
        token_probabilities = []
        for sentence_index in range(len(masked_sentences)):
            position = pos_masks[sentence_index][-1]
            sentence = masked_sentences[sentence_index]
            encoded_inputs = enc(
                sentence,
                return_tensors='pt', padding='max_length', max_length=Config.max_sentence_length + 1)
            output = mlm_model_ts(**encoded_inputs)[0][0][position]
            next_token_probability = torch.sum(torch.sort(F.softmax(output, dim=0))[0][-Config.possible_options:])
            token_probabilities.append(next_token_probability)
        matching_sentences = tuple(sentence for sentence, probability in zip(original_sentences, token_probabilities) if
                                   probability > Config.confidence_threshold)
        if not matching_sentences:
            return previous_match
        previous_match = matching_sentences
