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
model_name = 'bert-base-uncased'
mlm_model_ts = BertForMaskedLM.from_pretrained(model_name, torchscript=True)
tokenizer = BertTokenizer.from_pretrained(model_name)
traced_mlm_model = torch.jit.trace(mlm_model_ts, [tokens_tensor, segments_tensor, attention_masks_tensor])
mlm_model_ts.eval()
stripped_signs = '.?!,'


@torch.no_grad
def get_longest_masks(base_sentence: str) -> tuple[list[str], ...]:
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
            sentence = masked_sentences[sentence_index]
            tokenized_masked_positions = []
            tokenized_counter = 0
            word_to_tokens = {}
            for word_index in range(len(sentence)):
                word = sentence[word_index]
                if word not in word_to_tokens:
                    word_to_tokens[word] = tokenizer.tokenize(word)
                if word != Config.mask:
                    tokenized_counter += len(word_to_tokens[word])
                else:
                    word = base_sentence.split()[word_index]
                    if word not in word_to_tokens:
                        word_to_tokens[word] = tokenizer.tokenize(word)
                    for _ in word_to_tokens[word]:
                        tokenized_counter += 1
                        tokenized_masked_positions.append(tokenized_counter)
            pass
            for position in tokenized_masked_positions:
                encoded_inputs = enc(
                    ' '.join(sentence),
                    return_tensors='pt', padding='max_length', max_length=Config.max_sentence_length + 1)
                output = mlm_model_ts(**encoded_inputs)[0][0][position]
                next_token_probability = torch.sum(torch.sort(F.softmax(output, dim=0))[0][-Config.possible_options:])
                if next_token_probability < Config.confidence_threshold:
                    token_probabilities.append(next_token_probability)
                    break
            else:
                token_probabilities.append(next_token_probability)
        matching_sentences = tuple(sentence for sentence, probability in zip(original_sentences, token_probabilities) if
                                   probability > Config.confidence_threshold)
        if not matching_sentences:
            return previous_match
        previous_match = matching_sentences


def predict_from_mask(sentence: str):
    encoded_inputs = enc(
        sentence,
        return_tensors='pt', padding='max_length', max_length=Config.max_sentence_length + 1)
    while '[MASK]' in sentence.split():
        position = sentence.split().index('[MASK]') + 2
        output = mlm_model_ts(**encoded_inputs)[0][0][position]
        predicted_index = torch.argmax(output, dim=-1).item()
        predicted_token = enc.convert_ids_to_tokens([predicted_index])[0]
        sentence = sentence.replace('[MASK]', predicted_token, 1)
