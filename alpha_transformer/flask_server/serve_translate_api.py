from flask import Flask, request, jsonify
from scripts.infer import load_checkpoint_and_tokenizer, translate_sentences_non_batched
import argparse

# Initialize Flask app
app = Flask(__name__)

# Global reference to the loaded Trainer object
trainers = {}

# Define translation route
@app.route('/translate', methods=['POST'])
def translate():
    # parse json request
    data = request.get_json()
    sentences = data.get('text')
    target_language = data.get('target_language', 'fr')
    decode_type = data.get('decoder_type', 'beam')
    beam_size = int(data.get('beam_size', 5))
    
    # handle missing input
    if not sentences:
        return jsonify({'error': 'no input text provided'}), 400
    
    # Choose the correct model
    trainer = trainers.get(target_language)
    if not trainer:
        return jsonify({'error': f"No model available for target language '{target_language}'"}), 400
    
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
    global trainers

    # CLI arguments
    parser = argparse.ArgumentParser(description="Serve translation API from a trained Transformer model.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    args = parser.parse_args()

    # load model and tokenizer from checkpoint
    print("Loading models...")
    trainers['fr'] = load_checkpoint_and_tokenizer("checkpoints/en_fr_large_512_long/best_model.pt")
    trainers['de'] = load_checkpoint_and_tokenizer("checkpoints/en_de_run1/en_de_small_d128_v16k_20250615_050243/best_model.pt")
    print("Models loaded.")

    # start Flask server
    app.run(host="0.0.0.0", port=args.port)

# Execute only if run as a script
if __name__ == "__main__":
    main()
