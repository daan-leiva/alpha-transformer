import sacrebleu
import torch

class BLEUScorer:
    def __init__(self, tokenizer, eos_token_id, pad_token_id):
        """
        Args:
            tokenizer: SentencePiece tokenizer or any tokenizer with decode_ids method.
            eos_token_id: ID of the end-of-sequence token.
            pad_token_id: ID of the padding token.
        """
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def decode(self, ids):
        """
        Converts a sequence of token IDs to a decoded sentence.
        Stops at the first EOS token and removes padding tokens.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Remove pad tokens
        ids = [i for i in ids if i != self.pad_token_id]

        # Cut at EOS if it exists
        if self.eos_token_id in ids:
            ids = ids[:ids.index(self.eos_token_id)]

        # Decode using tokenizer
        return self.tokenizer.decode_ids(ids)

    def get_predictions_and_references(self, output, tgt_output):
        """
        Converts model logits and ground truth IDs to decoded sentences.
        Args:
            output: Tensor of shape (batch_size, seq_len, vocab_size)
            tgt_output: Tensor of shape (batch_size, seq_len)
        Returns:
            (predictions, references): Both are lists of decoded strings.
        """
        # Get most likely token from logits
        predicted_ids = output.argmax(dim=-1)  # (batch_size, seq_len)

        # Decode each sequence
        preds = [self.decode(pred) for pred in predicted_ids]
        refs = [self.decode(ref) for ref in tgt_output]

        return preds, refs

    def compute_corpus_bleu(self, predictions, references):
        """
        Computes corpus-level BLEU using sacrebleu.
        Args:
            predictions (List[str]): List of hypothesis sentences.
            references (List[str]): List of reference sentences.
        Returns:
            BLEU score (float)
        """
        # sacrebleu expects references as a list of reference lists: [[ref1, ref2, ...]]
        return sacrebleu.corpus_bleu(predictions, [references], smooth_method='exp').score
