import torch
from utils.bleu_scorer import BLEUScorer
import time
from utils.early_stopping import EarlyStopping
from tqdm import tqdm
from utils.training_plotter import TrainingPlotter
import argparse
import sentencepiece as spm

class Trainer:
    # Different use cases and required arguments:
    #
    # Training requires:
    #   - model
    #   - optimizer
    #   - scheduler
    #   - loss_fn
    #   - train_loader
    #   - val_loader
    #   - tokenizer (e.g., SentencePieceProcessor)
    #   - special_tokens (dictionary with pad, sos, eos)
    #   - args (Namespace or dict with training config)
    #   - log_file (for logging training output)
    #
    # Validation requires:
    #   - model
    #   - loss_fn
    #   - val_loader
    #   - tokenizer
    #   - special_tokens
    #   - args
    #   - log_file
    #
    # Inference requires:
    #   - model
    #   - tokenizer
    #   - special_tokens
    #   - args

    def __init__(self, model, tokenizer, args=None, special_tokens=None, optimizer=None,
                 scheduler=None, loss_fn=None, train_loader=None,
                 val_loader=None, log_file=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.tokenizer = tokenizer
        # create the bleu scorer using the tokenizer
        if special_tokens:
            self.bleu_scorer = BLEUScorer(tokenizer=self.tokenizer,
                                      eos_token_id=special_tokens['<eos>'],
                                      pad_token_id=special_tokens['<pad>'])
        
        # check args type (could be dict or args)
        if isinstance(args, dict):
            self.args = argparse.Namespace(**args)
        else:
            self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_file = log_file
        self.current_epoch = 0

        self.train_losses = []
        self.val_losses = []
        self.val_bleu_scores = []
        self.best_bleu_score = float('-inf')

        self.special_tokens=special_tokens
        self.overall_start_time = time.time()

        # print args
        if self.args:
            self.log_and_print("\nParsed Arguments:")
            for key, value in vars(self.args).items():
                self.log_and_print(f"{key}: {value}")
            self.log_and_print("-" * 50)

        # print cuda availability
        self.log_and_print("Cuda available: " + str(torch.cuda.is_available())) 

    # UTILITY FUNCTIONS

    # creates a padding mask for the source data
    # src shape : (batch_size, src_len)
    @staticmethod
    def create_src_mask(src, pad_token_id):
         # Ensure input is a torch.Tensor
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"Expected src to be a torch.Tensor, but got {type(src)}")

        # Ensure src is 2-dimensional: (batch_size, src_len)
        if src.ndim != 2:
            raise ValueError(f"Expected src to have 2 dimensions (batch_size, src_len), but got shape {src.shape}")

        return (src !=pad_token_id).unsqueeze(1).unsqueeze(1) # (batch_size,1,1,src_len) [broadcasting shape]

    # creata a padding mask for the target data
    # combines a causal look ahead mask with a padding token mask
    # tgt shape : (batch_size, targtet_len)
    @staticmethod
    def create_tgt_mask(tgt, pad_token_id):
        # Check that tgt is a tensor
        if not isinstance(tgt, torch.Tensor):
            raise TypeError(f"Expected tgt to be a torch.Tensor, but got {type(tgt)}")

        # Check that tgt has 2 dimensions: (batch_size, tgt_len)
        if tgt.dim() != 2:
            raise ValueError(f"Expected tgt to have 2 dimensions (batch_size, tgt_len), but got shape {tgt.shape}")

        # Ensure pad_token_id is an integer
        if not isinstance(pad_token_id, int):
            raise TypeError(f"pad_token_id should be an integer, but got {type(pad_token_id)}")
        
        # process mask
        tgt_len = tgt.shape[1]
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt.device).unsqueeze(0).unsqueeze(0) # (tgt_len, tgt_len)
        padding_mask = (tgt!=pad_token_id).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, tgt_len)
        result = causal_mask & padding_mask # should be of shape (batch, 1, tgt_len, tgt_len)
        # verify that the broadcast was done correctly
        assert(result.shape == (tgt.shape[0], 1, tgt_len, tgt_len))

        return result
    
    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time//3600)
        elapsed_mins = int((elapsed_time//3600)//60)
        elpased_secs = int(elapsed_time%60)

        return elapsed_hours, elapsed_mins, elpased_secs

    @staticmethod
    def estimate_remaining_time(start_time, elapsed_epochs, total_epochs):
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time/elapsed_epochs
        remaining_epochs = total_epochs - elapsed_epochs
        remaining_time = remaining_epochs * avg_epoch_time
        SECONDS_IN_MIN = 60
        SECONDS_IN_HOUR = SECONDS_IN_MIN * 60
        SECONDS_IN_DAY = SECONDS_IN_HOUR * 24

        rem_days = int(remaining_time // SECONDS_IN_DAY)
        remaining_time = remaining_time % SECONDS_IN_DAY

        rem_hrs = int(remaining_time // SECONDS_IN_HOUR)
        remaining_time = remaining_time % SECONDS_IN_HOUR

        rem_mins = int(remaining_time // SECONDS_IN_MIN)
        rem_secs = int(remaining_time % SECONDS_IN_MIN)

        return rem_days, rem_hrs, rem_mins, rem_secs

    @staticmethod
    def format_time(days, hours, minutes, seconds):
        parts = []
        if days > 0:
            parts.append(f'{days}d')
        if hours > 0 or days > 0:  # show hours if days > 0
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0 or days > 0:  # show minutes if hours/days exist
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")  # always show seconds

        return ' '.join(parts)    
    
    def save_checkpoint(self, path):
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': vars(self.args),
            'train_losses': self.train_losses,
            'val_losses' : self.val_losses,
            'val_bleu_scores' : self.val_bleu_scores,
            'best_bleu_score': self.best_bleu_score,
            'special_tokens' : self.special_tokens
        }
        torch.save(state, path)

    def load_checkpoint(self, path, need_optimizer=False, need_scheduler=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if need_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if need_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1  # resume from next epoch
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_bleu_scores = checkpoint['val_bleu_scores']
        self.best_bleu_score = checkpoint['best_bleu_score']
        self.args = argparse.Namespace(**checkpoint['args'])
        self.special_tokens = checkpoint['special_tokens']
        # create the bleu scorer based on the special_tokens
        self.bleu_scorer = BLEUScorer(tokenizer=self.tokenizer,
                                      eos_token_id=self.special_tokens['<eos>'],
                                      pad_token_id=self.special_tokens['<pad>'])
    
    def log_and_print(self, message):
        print(message)
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()

    # Train | Validate | Inference

    # one batch step in training
    def train_one_batch(self, src, tgt):
        # check that we have the needed training mode variables
        if self.optimizer is None or self.loss_fn is None:
            self.log_and_print("Trainer was intialized without optimize or loss_fn. Training unavailable")
            return
        # offset the target input/out
        tgt_input = tgt[:, :-1] # skip the last token
        tgt_output = tgt[:, 1:] # skip the first token
        
        # masks
        src_mask = self.create_src_mask(src, pad_token_id=self.special_tokens['<pad>'])
        tgt_mask = self.create_tgt_mask(tgt_input, pad_token_id=self.special_tokens['<pad>'])

        # forward pass through the model
        output = self.model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

        # clear the gradient
        self.optimizer.zero_grad()

        # reshape outputs
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        # calculate loss
        batch_loss = self.loss_fn(output, tgt_output)

        # calcualte the gradient
        batch_loss.backward()

        # update the weights
        self.optimizer.step()

        return batch_loss.item()
    
    # one batch step in training
    def validate_one_batch(self, src, tgt):
        # check that the correct variables were set for validation
        if self.loss_fn is None:
            self.log_and_print("Trainer was intialized without loss_fn. Validation unavailable")
            return
        # offset the target input/out
        tgt_input = tgt[:, :-1] # skip the last token
        tgt_output = tgt[:, 1:] # skip the first token
        
        # masks
        src_mask = self.create_src_mask(src, pad_token_id=self.special_tokens['<pad>'])
        tgt_mask = self.create_tgt_mask(tgt_input, pad_token_id=self.special_tokens['<pad>'])

        # forward pass through the model
        output = self.model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

        # reshape outputs
        # calculate loss
        batch_loss = self.loss_fn(output.reshape(-1, output.shape[-1]),
                            tgt_output.reshape(-1))

        return batch_loss.item(), output

    # train the model (this includes a validation run)
    def train(self):
        if self.args is None or self.train_loader is None or \
        self.valid_loader is None or self.scheduler is None or \
        self.loss_fn is None or self.optimizer is None:
            self.log_and_print("Please set all of the variables if training. Operation canceled")
            return
        # helper objects
        early_stopping = EarlyStopping(patience=9, min_delta=0.1, mode='max')
        epochs = self.args.num_epochs
        model_path = f'./checkpoints/{self.args.save_path}'
        # train loop
        for epoch in range(self.current_epoch, epochs):
            # this aids with saving/loading training state
            self.current_epoch = epoch
            # start timer
            epoch_start_time = time.time()
            # train model
            train_epoch_loss = 0
            self.model.train()
            for src_batch, tgt_batch in tqdm(self.train_loader, desc=F'Epoch {epoch+1:0{len(str(epochs))}}/{epochs} [Train]', leave=False):
                # get data for this batch
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)

                # train a single batch
                train_batch_loss = self.train_one_batch(src=src_batch, tgt=tgt_batch)
                # accumulate loss
                train_epoch_loss += train_batch_loss

            val_epoch_loss, bleu_score = self.validate()

            # calculate aggregate metrics
            train_epoch_loss /= len(self.train_loader)

            # append metrics
            self.train_losses.append(train_epoch_loss)
            self.val_losses.append(val_epoch_loss)
            self.val_bleu_scores.append(bleu_score)

            # stop time and calculate time metrics
            epoch_end_time = time.time()
            elapsed_hours, elapsed_mins, elpased_secs = self.epoch_time(epoch_start_time, epoch_end_time)
            rem_days, rem_hrs, rem_mins, rem_secs = self.estimate_remaining_time(self.overall_start_time, epoch + 1, epochs)
            elapsed_epoch_time_str = self.format_time(0, elapsed_hours, elapsed_mins, elpased_secs) # pass zero for days
            rem_time_str = self.format_time(rem_days, rem_hrs, rem_mins, rem_secs)
            # print epoch summary
            self.log_and_print(f'Epoch {epoch + 1:0{len(str(epochs))}}/{epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}' +
                f' | BLEU Score: {bleu_score:.2f} | Time: {elapsed_epoch_time_str} | ETA: {rem_time_str}')

            # save best model
            if bleu_score > self.best_bleu_score:
                self.best_bleu_score = bleu_score
                self.save_checkpoint(path=f'{model_path}/best_model.pt')
            # save every 5 epochs as well
            if epoch%5==0:
                self.save_checkpoint(path=f'{model_path}/checkpoint_epoch_{epoch}.pt')
            # check for early stopping
            if early_stopping.step(metric=bleu_score):
                self.log_and_print("Early Stopped")
                break

            self.scheduler.step()

        # plot results
        plotter = TrainingPlotter(training_losses=self.train_losses,
                                val_losses=self.val_losses,
                                val_bleu_scores=self.val_bleu_scores)
        #plotter.plot()
        plot_filename = f'{model_path}/training_curves.png'
        plotter.save(filename=plot_filename)

    def validate(self):
        if self.valid_loader is None or self.loss_fn is None:
            raise ValueError("Missing val_loader or loss_fn for validation.")   
        val_loss = 0
        all_preds = []
        all_refs = []
        self.model.eval()

        with torch.no_grad():
            for src_batch, tgt_batch in tqdm(self.valid_loader, desc=f" [Validation]", leave=False):
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)

                batch_loss, output = self.validate_one_batch(src_batch, tgt_batch)

                preds, refs = self.bleu_scorer.get_predictions_and_references(output, tgt_batch[:, 1:])
                all_preds.extend(preds)
                all_refs.extend(refs)

                val_loss += batch_loss

        val_loss /= len(self.valid_loader)
        bleu_score = self.bleu_scorer.compute_corpus_bleu(all_preds, all_refs)

        return val_loss, bleu_score
    
    # expects a torch tensor as an input
    def infer(self, src, decode_type='beam', beam_size=None, return_attention=False):
        # === Input Checks ===
        if not torch.is_tensor(src):
            raise TypeError(f"`src` must be a torch.Tensor, got {type(src)}")
        if src.ndim != 2:
            raise ValueError(f"`src` must be a 2D tensor (batch_size, seq_len), got shape {src.shape}")
        if decode_type not in ['beam', 'greedy']:
            raise ValueError(f"`decode_type` must be either 'beam' or 'greedy', got {decode_type}")
        if beam_size is not None and (not isinstance(beam_size, int) or beam_size <= 0):
            raise ValueError(f"`beam_size` must be a positive integer, got {beam_size}")
        
        # greedly get sequence ids
        self.model.eval()
        if decode_type == 'greedy':
            if return_attention:
                generated_sequences, final_attn = self.greedy_decode(src, return_attention=return_attention)
            else:
                generated_sequences =  self.greedy_decode(src)
        elif decode_type == 'beam':
            if return_attention:
                if beam_size:
                    generated_sequences, final_attn = self.beam_search_decode(src, beam_size=beam_size, return_attention=return_attention)
                    
                else:
                    generated_sequences, final_attn = self.beam_search_decode(src, return_attention=return_attention)
            else:
                if beam_size:
                    generated_sequences = self.beam_search_decode(src, beam_size=beam_size)
                else:
                    generated_sequences= self.beam_search_decode(src)
        else:
            raise ValueError("Inference decode_type can only be greedy or beam")
        # get the sentences in text
        decoded_sentences = self.decode_ids(id_sequences=generated_sequences)
        if return_attention:
            return decoded_sentences, generated_sequences, final_attn
        else:
            return decoded_sentences, generated_sequences

    def decode_ids(self, id_sequences, sos_token='<sos>', eos_token='<eos>', pad_token='<pad>'):
        # check if the sequences are tensor type
        if isinstance(id_sequences, torch.Tensor):
            # move to cpu for processing
            id_sequences = id_sequences.cpu().tolist()
        sentences = []
        # get token id
        eos_id = self.special_tokens[eos_token]
        sos_id = self.special_tokens[sos_token]
        pad_id = self.special_tokens[pad_token]
        # iterate through each sentence in the batch
        for seq in id_sequences:
            # cut off at first eos
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            # remove sos and pad
            seq = [id for id in seq if id not in (sos_id, pad_id)]
            # decode
            sentence = self.tokenizer.decode_ids(seq)
            sentences.append(sentence)

        return sentences
    
    def greedy_decodeOLD(self, src):
        src = src.to(self.device)
        batch_size = src.size(0)
        sos_token_id = self.special_tokens['<sos>']
        eos_token_id = self.special_tokens['<eos>']
        pad_token_id = self.special_tokens['<pad>']
        max_len = self.args.max_len

        # start with the <sos> token for each sentene in the batch
        generated = torch.full((batch_size, 1), fill_value=sos_token_id,
                               dtype=torch.long, device=self.device)
        # create the src maks
        src_mask = self.create_src_mask(src, pad_token_id=pad_token_id)

        # to keep track of finished sequences
        # all initialized to false
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # keep going until max_len or all have an eos
            for _ in range(max_len):
                tgt_mask = self.create_tgt_mask(generated, pad_token_id=pad_token_id)

                # predict logits
                # shape (batch_size, seq_len, vocab_size)
                outputs = self.model(src=src, tgt=generated, src_mask=src_mask,
                                    tgt_mask=tgt_mask)
                
                # take the last value of the sequence so far
                # shape (batch_size, vocab size)
                next_token_logits = outputs[:, -1, :]

                # take the most likely vocab word
                # shape (batch_size)
                next_tokens = next_token_logits.argmax(dim=-1)

                # append next tokens
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

                # update finished sequences
                # this keeps it as finished or updates if the eos
                # token is there
                finished = finished | next_tokens == eos_token_id

                # break out of the loop if all done
                if finished.all():
                    break

        return generated
    
    # expects a torch tensor as an input
    def greedy_decode(self, src, return_attention=False, debug_mode=False):
        # === Input validation ===
        if not torch.is_tensor(src):
            raise TypeError(f"`src` must be a torch.Tensor, got {type(src)}")
        if src.ndim != 2:
            raise ValueError(f"`src` must be a 2D tensor of shape (batch_size, seq_len), got {src.shape}")
        
        # needed variables
        src = src.to(self.device)
        batch_size = src.size(0)
        sos_token_id = self.special_tokens['<sos>']
        eos_token_id = self.special_tokens['<eos>']
        pad_token_id = self.special_tokens['<pad>']
        max_len = self.args.max_len

        # creta the src maks
        src_mask = self.create_src_mask(src, pad_token_id=pad_token_id)

        # create a variable for the past key values to reuse
        # will be none for the first token
        past_key_values = None

        # create an attention matrix if needed
        attention_matrices = [] if return_attention else None

        with torch.no_grad():
            # get the encoder output to be reused
            memory = self.model.encode(src=src, src_mask=src_mask)
            # start with the <sos> token for each sentence in the batch
            generated = torch.full((batch_size, 1), fill_value=sos_token_id,
                                dtype=torch.long, device=self.device)
            # to keep track of finished sequences
            # all initialized to false
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            # keep going until max_len or all have an eos
            for _ in range(max_len):
                decoder_input = generated[:, -1:]
                # predict logits
                # shape (batch_size, seq_len, vocab_size)
                result = self.model.decode(tgt=decoder_input, encoder_output=memory,
                                           src_mask=src_mask, past_key_values=past_key_values,
                                           return_attention=return_attention)
                if return_attention:
                    decoder_output, past_key_values, cross_attention = result
                    attention_matrices.append(cross_attention)
                else:
                    decoder_output, past_key_values = result
                vocab_logits = self.model.vocab_projection(decoder_output)
                
                # take the last value of the sequence so far
                # shape (batch_size, vocab size)
                next_token_logits = vocab_logits[:, -1, :]

                # take the most likely vocab word
                # shape (batch_size)
                next_tokens = next_token_logits.argmax(dim=-1)

                # append next tokens
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

                # update finished sequences
                # this keeps it as finished or updates if the eos
                # token is there
                finished = finished | next_tokens == eos_token_id

                # break out of the loop if all done
                if finished.all():
                    break
         # Convert generated tensor to list of lists of ints
        generated_sequences = [seq.tolist() for seq in generated]

        if return_attention:
            if debug_mode:
                print("attn shape before stack: ", len(attention_matrices), ', ', attention_matrices[0].shape)
            # attention shape before processing:
            # len() = steps  tensor_shape = (num_layers, batch_size, num_heads, 1, src_len)
            attention_tensor = torch.stack(attention_matrices, dim=0)
            # after stack the shape is (steps|tgt_len, num_layers, batch_size, num_heads, 1, src_len)
            # squeeze singleton out
            attention_tensor = attention_tensor.squeeze(4)
            # Final shape after permute: (tgt_len, num_layers, batch_size, num_heads, src_len)
            attention_tensor = attention_tensor.permute(2, 1, 3, 0, 4)
            return generated_sequences, attention_tensor

        return generated_sequences
    
    def reorder_past_key_values(self, past_key_values, beam_indices):
        new_past = []
        # iterate each layer
        for layer_past in past_key_values:
            new_layer = []
            # then we reorder the list of key,values for that layer 
            for past_state in layer_past:
                # past_state shape: (batch_size * beam_size, n_heads, seq_len, head_dim)
                # sort the past_keys by beam_indices
                new_past_state = past_state.index_select(0, beam_indices)
                new_layer.append(new_past_state)
            new_past.append(tuple(new_layer))
        return new_past

    
    # expects a torch tensor as an input
    @torch.no_grad()
    def beam_search_decode(self, src, beam_size=5, length_penalty_alpha=2,
                           repetition_penalty=20,
                           return_attention=False, debug_mode=False):
        # === Input validation ===
        if not torch.is_tensor(src):
            raise TypeError(f"`src` must be a torch.Tensor, got {type(src)}")
        if src.ndim != 2:
            raise ValueError(f"`src` must be a 2D tensor of shape (batch_size, seq_len), got {src.shape}")
        
        # useful variables
        src = src.to(self.device)
        batch_size = src.size(0)

        sos_token_id = self.special_tokens['<sos>']
        eos_token_id = self.special_tokens['<eos>']
        pad_token_id = self.special_tokens['<pad>']
        max_len = self.args.max_len

        # prepare src
        src_mask = self.create_src_mask(src, pad_token_id=pad_token_id)
        if debug_mode:
            print("Src shape: ", src_mask.shape)
            print_done = False
        memory = self.model.encode(src, src_mask=src_mask)

        # repeat for each beam
        memory = memory.repeat_interleave(beam_size, dim=0)
        src_mask = src_mask.repeat_interleave(beam_size, dim=0)

        # intialize beams
        generated_sequences = torch.full(size=(batch_size * beam_size, 1),
                               fill_value=sos_token_id, dtype=torch.long,
                               device=self.device)
        
        # beam scores
        beam_log_probs = torch.zeros((batch_size, beam_size), device=self.device)
        beam_log_probs[:, 1:] = float('-inf') # only first beam active
        beam_log_probs = beam_log_probs.view(-1) # flatten to (batch_size * beam_size,)

        #  tracking finished sequences
        is_beam_finished = torch.zeros(size=(batch_size, beam_size), dtype=torch.bool, device=self.device)

        # caching for key values
        past_key_values = None

        # create an attention matrix if needed
        if return_attention:
            all_cross_attn = []

        # iterate through the length (or until all sequences are done)
        for step in range(max_len):
            # decode the last token (since we are using cashing)
            decoder_input = generated_sequences[:, -1:]

            # don't need a source mask since only one value is being
            # passed at a time
            if return_attention:
                decoder_output, past_key_values, cross_attention = self.model.decode(
                    tgt=decoder_input, encoder_output=memory,
                    src_mask=src_mask,
                    tgt_mask=None,
                    past_key_values=past_key_values,
                    return_attention=return_attention
                )
                if debug_mode:
                    if not print_done:
                        print("Cross attention shape: ", cross_attention.shape)
                        print_done = True
                all_cross_attn.append(cross_attention)
            else:
                decoder_output, past_key_values = self.model.decode(
                    tgt=decoder_input, encoder_output=memory,
                    src_mask=src_mask,
                    tgt_mask=None,
                    past_key_values=past_key_values
                )

            vocab_logits = self.model.vocab_projection(decoder_output)

            # from the seq_len dimension extract the last token
            current_step_logits = vocab_logits[:, -1, :]  # (batch_size * beam_size, vocab_size)

            # get log probs (for numerically stability)
            next_token_log_probs = torch.log_softmax(current_step_logits, dim=-1)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(next_token_log_probs.size(0)):
                    for token in set(generated_sequences[i].tolist()):
                        if next_token_log_probs[i, token] < 0:
                            next_token_log_probs[i, token] *= repetition_penalty
                        else:
                            next_token_log_probs[i, token] /= repetition_penalty

            # add precious beam scores (expand to broadcast)
            # next_token_log_probs shape (batch_size * beam_size, vocab_size)
            # beam_scores shape (batch_size * beam_size) [add dim here]
            # result will be (batch_size * beam_size, vocab_size)
            # so that we have a score per potention output
            candidate_log_probs = next_token_log_probs + beam_log_probs.unsqueeze(-1)

            # reshape so the last dimension is beam_size*vocab_size
            candidate_log_probs = candidate_log_probs.view(batch_size, -1)

            # get top k beams from the beam_size*vocab_size
            # returns values and indexes of the topk scores
            # (values, indices)
            # top_k
            topk_log_probs, topk_condidate_indices = candidate_log_probs.topk(k=beam_size, dim=1,
                                                    largest=True, sorted=True)
            
            # decode the flattened candidates indices
            selected_beam_indices = topk_condidate_indices // next_token_log_probs.size(-1) # which beam it came from
            selected_token_ids = topk_condidate_indices % next_token_log_probs.size(-1)

            # compute flat indices for gathering previous sequences
            batch_beam_offset = torch.arange(end=batch_size, device=self.device) * beam_size
            # add the offset
            flat_selected_beam_indices = (selected_beam_indices + batch_beam_offset.unsqueeze(1)).view(-1)

            # gather and update sequences
            # extract the selected beams
            generated_sequences = generated_sequences[flat_selected_beam_indices]

            # mask out finished beams: once EOS is generated, force pad token
            flat_is_finished = is_beam_finished.view(-1)
            selected_token_ids = selected_token_ids.view(-1)

            # append next token
            selected_token_ids[flat_is_finished] = pad_token_id
            generated_sequences = torch.cat(
                [generated_sequences, selected_token_ids.view(-1, 1)], dim=-1
            )

            # reorder past key values if caching
            if past_key_values is not None:
                past_key_values = self.reorder_past_key_values(past_key_values=past_key_values,
                                                              beam_indices=flat_selected_beam_indices)

            # update attention matrix if needed
            if return_attention:
                # reorder the current attention matrix to match the beam order
                reordered_cross_attn = []
                for layer_attn in cross_attention:
                    reordered_layer = layer_attn[flat_selected_beam_indices]
                    reordered_cross_attn.append(reordered_layer)
                # replace current attention matrx
                all_cross_attn[-1] = reordered_cross_attn  
                
            # update beam scores
            beam_log_probs = topk_log_probs.view(-1)

            # track beams that have generated <eos> and mark them as finished
            is_eos_token = (selected_token_ids == eos_token_id)
            # by using an or operator we keep the htings that were marked as finished
            is_beam_finished = is_beam_finished | is_eos_token

            # early stopping if all beams are done
            if is_beam_finished.all():
                break
        
        if debug_mode:
            print("Final step count: ", step)

        # reshape back to (batch_size, beam_size, seq_len)
        generated_sequences = generated_sequences.view(batch_size, beam_size, -1)

        # length normalization
        # take the values where they are not padded. Sum across the seq length and
        # convert it to a float (returns batch_size, beam_size) obejct
        sequence_lengths = (generated_sequences != pad_token_id).sum(dim=-1).float()
        # reshape beam log probabiliies to (batch_size, beam_size)
        normalized_scores = beam_log_probs.view(batch_size, beam_size) / ((sequence_lengths ** length_penalty_alpha))
        if debug_mode:
            print("Raw beam log probs:", beam_log_probs.view(batch_size, beam_size))
            print("Sequence lengths:", sequence_lengths)
            print("Normalized scores:", normalized_scores)


        # select the beat beam
        best_beam_idx = normalized_scores.argmax(dim=-1)

        # gather the best sequences
        best_sequences = []
        for i in range(batch_size):
            best_seq = generated_sequences[i, best_beam_idx[i]].tolist()
            best_sequences.append(best_seq)

        # Truncate at first </s> and strip pads before it
        truncated_sequences = []
        for seq in best_sequences:
            try:
                eos_idx = seq.index(eos_token_id)
            except ValueError:
                eos_idx = len(seq)
            cleaned_seq = [tok for tok in seq[:eos_idx+1] if tok != pad_token_id]
            truncated_sequences.append(cleaned_seq)
        best_sequences = truncated_sequences

        if return_attention:
            # Transpose: Convert from list[tgt_len] of list[num_layers] → list[num_layers] of list[tgt_len]
            # This groups attention matrices by layer instead of by time step
            if debug_mode:
                print('Attention shape before going in: ', len(all_cross_attn), ', ',
                      len(all_cross_attn[0]), ', ', all_cross_attn[0][0].shape)
            layerwise = list(zip(*all_cross_attn))

            # For each layer, concatenate attention matrices across time steps
            # Each `layer` is a list of tensors of shape (num_heads, batch*beam, 1, src_len) — one per time step
            # Resulting tensor shape: (batch*beam, num_heads, tgt_len, src_len) for each layer
            stacked = [torch.cat(layer, dim=2) for layer in layerwise]
            if debug_mode:
                print('Stacked dimension: ', len(stacked), ', ', stacked[0].shape)

            # Stack all layers into a single tensor
            # Shape: (num_layers, batch*beam, num_heads, tgt_len, src_len)
            cross_attn_all_layers = torch.stack(stacked, dim=0)

            # Compute flat indices for the best beam in each batch
            # This is needed to extract the attention maps for just the best hypotheses
            selected_indices = (best_beam_idx + batch_beam_offset).tolist()

            # Index into the stacked attention to get just the best beam for each batch
            # Final shape: (num_layers, batch_size, num_heads, tgt_len, src_len)
            final_attn = cross_attn_all_layers[:, selected_indices, :, :, :]

            # reorder to have batch first
            final_attn = final_attn.permute(1, 0, 2, 3, 4)
            if debug_mode:
                print("final_attn.shape =", final_attn.shape)

            return best_sequences, final_attn

        return best_sequences
