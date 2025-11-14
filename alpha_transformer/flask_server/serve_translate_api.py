from flask import Flask, request, jsonify
from scripts.infer import load_checkpoint_and_tokenizer, translate_sentences_non_batched
import argparse

# Initialize Flask app
app = Flask(__name__)

# Models are stored here keyed by target language code, for example "fr" or "de"
trainers = {}

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate a sentence or list of sentences using a loaded model.

    Expected JSON payload:
        {
            "text": "one sentence" or ["s1", "s2", ...],
            "target_language": "fr" or "de",
            "decoder_type": "beam" or "greedy",
            "beam_size": 5
        }
    """
    data = request.get_json()

    # Input can be a single string or a list of strings
    sentences = data.get('text')
    # Default to French
    target_language = data.get('target_language', 'fr')
    # Decoding method: greedy or beam
    decode_type = data.get('decoder_type', 'beam')
    # Beam width, defaults to 5
    beam_size = int(data.get('beam_size', 5))

    # Input can be a single string or a list of strings
    if not sentences:
        return jsonify({'error': 'no input text provided'}), 400

    # Look up trainer for the requested target language
    trainer = trainers.get(target_language)
    if not trainer:
        return jsonify({'error': f"No model available for target language '{target_language}'"}), 400

    # Normalize input to a list
    if isinstance(sentences, str):
        sentences = [sentences]

    # Run inference and request attention for visualization
    results, input_tokens, output_tokens, attentions = translate_sentences_non_batched(
        trainer,
        sentences,
        decode_type,
        beam_size,
        return_attention=True
    )

    # Convert attention tensors to plain lists for JSON serialization
    processed_attentions = [attn_tensor.tolist() for attn_tensor in attentions]

    # Return full response: translation, tokenized inputs/outputs, and attention maps
    return jsonify({
        'translation': results,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'attentions': processed_attentions
    })


def main():
    """
    Entry point for running the Flask app as a script.

    Loads the pretrained models for French and German and starts the server.
    """
    global trainers

    # Parse CLI arguments (e.g., --port 6000)
    parser = argparse.ArgumentParser(description="Serve translation API from a trained Transformer model.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    args = parser.parse_args()

    # Load trained models and their tokenizers
    print("Loading models...")
    trainers['fr'] = load_checkpoint_and_tokenizer("checkpoints/en_fr_large_512_long/best_model.pt")
    trainers['de'] = load_checkpoint_and_tokenizer("checkpoints/en_de_run1/en_de_medium_d256_v8k/best_model.pt")
    print("Models loaded.")

    # For production, wrap this app with gunicorn or another WSGI server
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()