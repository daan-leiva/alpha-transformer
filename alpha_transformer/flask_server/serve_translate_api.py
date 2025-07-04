from flask import Flask, request, jsonify
from scripts.infer import load_checkpoint_and_tokenizer, translate_sentences_non_batched
import argparse

# === Initialize Flask app ===
app = Flask(__name__)

# === Global dictionary to store loaded models (trainer objects) for each target language ===
trainers = {}

# === Define POST endpoint for translation ===
@app.route('/translate', methods=['POST'])
def translate():
    # Parse the incoming JSON payload
    data = request.get_json()
    sentences = data.get('text')  # Input can be a single string or a list of strings
    target_language = data.get('target_language', 'fr')  # Default to French
    decode_type = data.get('decoder_type', 'beam')       # Decoding method: greedy or beam
    beam_size = int(data.get('beam_size', 5))            # Beam width, defaults to 5

    # Handle missing input
    if not sentences:
        return jsonify({'error': 'no input text provided'}), 400

    # Look up the correct trainer (model + tokenizer) for the requested target language
    trainer = trainers.get(target_language)
    if not trainer:
        return jsonify({'error': f"No model available for target language '{target_language}'"}), 400

    # Ensure input is a list (even if it's a single string)
    if isinstance(sentences, str):
        sentences = [sentences]

    # Run inference using the loaded model and return attention matrices
    results, input_tokens, output_tokens, attentions = translate_sentences_non_batched(
        trainer,
        sentences,
        decode_type,
        beam_size,
        return_attention=True
    )

    # Convert attention matrices (tensors) into lists so they can be JSON-serialized
    processed_attentions = [attn_tensor.tolist() for attn_tensor in attentions]

    # Return full response: translation, tokenized inputs/outputs, and attention maps
    return jsonify({
        'translation': results,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'attentions': processed_attentions
    })


# === Entry point to start the Flask app ===
def main():
    global trainers

    # Parse CLI arguments (e.g., --port 6000)
    parser = argparse.ArgumentParser(description="Serve translation API from a trained Transformer model.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    args = parser.parse_args()

    # Load trained models and their associated tokenizers from disk
    print("Loading models...")
    trainers['fr'] = load_checkpoint_and_tokenizer("checkpoints/en_fr_large_512_long/best_model.pt")
    trainers['de'] = load_checkpoint_and_tokenizer("checkpoints/en_de_run1/en_de_medium_d256_v8k_20250615_050243/best_model.pt")
    print("Models loaded.")

    # Start the Flask development server (use gunicorn in production)
    app.run(host="0.0.0.0", port=args.port)


# === Run the main function only if the script is executed directly ===
if __name__ == "__main__":
    main()
