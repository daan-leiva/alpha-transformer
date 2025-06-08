import sacrebleu
import torch
        
class BLEUScorer:
    def __init__(self, tokenizer, eos_token_id, pad_token_id):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer

    # returns a list of decoded tokens
    def decode(self, ids):
        # convert if passed as tensor
        ids = [token_id.item() if isinstance(token_id, torch.Tensor) else token_id for token_id in ids]
        # remove padding
        ids = [token_id for token_id in ids if token_id != self.pad_token_id]

        # end at eos
        if self.eos_token_id in ids:
            ids = ids[:ids.index(self.eos_token_id)]

        # decode into sentences
        sentence = self.tokenizer.decode_ids(ids)

        return sentence
    
    def get_predictions_and_references(self, output, tgt_output):
        # get the max id per probability distribution
        output = output.argmax(dim=-1)

        # get predictions and references
        preds = [self.decode(pred) for pred in output]
        refs = [self.decode(ref) for ref in tgt_output]

        return preds, refs
    
    def compute_corpus_bleu(self, predictions, references):
        # sacrebleu requires a list of references
        references = [references]
        bleu = sacrebleu.corpus_bleu(predictions, references, smooth_method='exp')
        return bleu.score