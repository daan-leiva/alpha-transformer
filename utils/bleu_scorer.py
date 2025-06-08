import sacrebleu
import torch
        
class BLEUScorer:
    def __init__(self, tokenizer, pad_token_id=0, pad_token='<pad>', eos_token='<eos>'):
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.tokenizer = tokenizer

    # returns a list of decoded tokens
    def decode(self, ids):
        tokens = []
        for id in ids:
            word = self.id2word.get(id.item(), '<unk>')
            if word == self.eos_token:
                break
            if word != self.pad_token:
                tokens.append(word)
        return tokens
    
    def get_predictions_and_references(self, output, tgt_output):
        # get the max id per probability distribution
        output = output.argmax(dim=-1)

        # get predictions and references
        preds = [self.detokenize(self.decode(pred)) for pred in output]
        refs = [self.detokenize(self.decode(ref)) for ref in tgt_output]

        return preds, refs
    
    def compute_corpus_bleu(self, predictions, references):
        # sacrebleu requires a list of references
        references = [references]
        bleu = sacrebleu.corpus_bleu(predictions, references, smooth_method='exp')
        return bleu.score