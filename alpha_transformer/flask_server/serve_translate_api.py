from flask import Flask, request, jsonify
from scripts.infer import load_checkpoint_and_tokenizer, translate_sentences_non_batched
import argparse

# Initialize Flask app
app = Flask(__name__)

# Global reference to the loaded Trainer object
trainer = None

# Define translation route
@app.route('/translate', methods=['POST'])
def translate():
    # parse json request
    data = request.get_json()
    sentences = data.get('text')
    decode_type = data.get('decoder_type', 'beam')
    beam_size = int(data.get('beam_size', 5))

    # handle missing input
    if not sentences:
        return jsonify({'error': 'no input text provided'}), 400
    
    # ensure sentence is a list
    if isinstance(sentences, str):
        sentences = [sentences]

    # generate translations
    results, input_tokens, output_tokens, attentions = translate_sentences_non_batched(trainer, sentences,
                                                                           decode_type, beam_size,
                                                                           return_attention=True)
    
    # convert attention tensors to JSON-serializable format
    processed_attentions = [attn_tensor.tolist() for attn_tensor in attentions]

    return jsonify({'translation':results,
                    'input_tokens':input_tokens,
                    'output_tokens':output_tokens,
                    'attentions':processed_attentions})

# main entry point
def main():
    global trainer

    # CLI arguments
    parser = argparse.ArgumentParser(description="Serve translation API from a trained Transformer model.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt checkpoint")

    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    args = parser.parse_args()

    # load model and tokenizer from checkpoint
    trainer = load_checkpoint_and_tokenizer(args.checkpoint)

    # start Flask server
    app.run(host="0.0.0.0", port=args.port)

# Execute only if run as a script
if __name__ == "__main__":
    main()
