import sacrebleu
import torch

class BLEUScorer:
    """
    Helper for decoding token ids and computing corpus level BLEU scores.

    This class handles the mapping from model outputs to text and wraps
    sacrebleu so that evaluation code in Trainer can stay simple.
    """

    def __init__(self, tokenizer, eos_token_id, pad_token_id):
        """
        Parameters
        ----------
        tokenizer
            SentencePiece tokenizer or any object with a decode_ids method.
        eos_token_id : int
            Id of the end of sequence token.
        pad_token_id : int
            Id of the padding token.
        """
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def decode(self, ids):
        """
        Convert a sequence of token ids to a decoded sentence.

        Steps:
          convert tensors to plain lists
          remove pad tokens
          cut at the first eos token if present
          decode with the tokenizer
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Remove padding tokens
        ids = [i for i in ids if i != self.pad_token_id]

        # Cut at eos if present
        if self.eos_token_id in ids:
            ids = ids[:ids.index(self.eos_token_id)]

        # Decode using tokenizer
        return self.tokenizer.decode_ids(ids)

    def get_predictions_and_references(self, output, tgt_output):
        """
        Convert model logits and ground truth ids to decoded sentences.

        Parameters
        ----------
        output : Tensor
            Model logits of shape (batch_size, seq_len, vocab_size).
        tgt_output : Tensor
            Ground truth ids of shape (batch_size, seq_len).

        Returns
        -------
        tuple[list[str], list[str]]
            Two lists containing predicted sentences and reference sentences.
        """
        # Get most likely token from logits
        predicted_ids = output.argmax(dim=-1)  # (batch_size, seq_len)

        # Decode each sequence
        preds = [self.decode(pred) for pred in predicted_ids]
        refs = [self.decode(ref) for ref in tgt_output]

        return preds, refs

    def compute_corpus_bleu(self, predictions, references):
        """
        Compute corpus level BLEU using sacrebleu.

        Parameters
        ----------
        predictions : list[str]
            List of hypothesis sentences.
        references : list[str]
            List of reference sentences.

        Returns
        -------
        float
            BLEU score as returned by sacrebleu.
        """
        # sacrebleu expects references as a list of reference lists:
        # [ [ref_1, ref_2, ...] ]
        result = sacrebleu.corpus_bleu(
            predictions,
            [references],
            smooth_method="exp",
        )
        
        return result.score
