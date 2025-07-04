import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sentencepiece as spm

# === Parse metadata from checkpoint directory name ===
def parse_checkpoint_name(name):
    # Extract language, model size, d_model, and vocab size from folder name using regex
    match = re.match(r"en_(fr|de)_(small|medium|large)_d(\d+)_v(\d+)k", name)
    if match:
        lang, model_size, d_model, vocab_size = match.groups()
        return lang, model_size, int(d_model), int(vocab_size) * 1000
    return None, None, None, None

# === Load results from multiple checkpoints ===
def load_results(folder, skip_list):
    # Accumulate all model metadata and scores
    results = []

    for subdir in os.listdir(folder):
        if subdir in skip_list:
            print(f"[Skip] {subdir} is in exclusion list")
            continue

        path = os.path.join(folder, subdir)
        if not os.path.isdir(path):
            continue

        # Check for best model
        model_path = os.path.join(path, "best_model.pt")
        if not os.path.exists(model_path):
            print(f"[Missing] Skipping {subdir} - could not find best_model.pt")
            continue

        try:
            # Load checkpoint without using GPU
            checkpoint = torch.load(model_path, map_location="cpu")

            # Parse metadata from folder name
            args = checkpoint.get("args", {})
            lang, model_size, d_model, vocab_size = parse_checkpoint_name(subdir)
            bleu_score = checkpoint.get("best_bleu_score", -1)
            epoch = checkpoint.get("epoch", -1)

            # Load vocab size from SentencePiece model
            sp = spm.SentencePieceProcessor()
            sp.load(args['sp_model_path'])
            vocab_size = sp.get_piece_size()

            # Add results entry
            results.append({
                "Lang": lang,
                "Path": subdir,
                "Model Size": model_size,
                "D_model": d_model or args.get("d_model"),
                "Vocab Size": vocab_size,
                "Epoch": epoch,
                "BLEU Score": bleu_score
            })

        except Exception as e:
            print(f'[Error] Failed to read {subdir}: {e}')

    return pd.DataFrame(results)

# === Generate bar plot of BLEU scores ===
def plot_bleu_scores(df: pd.DataFrame, lang: str,
                     output_file=None,
                     show_plot=False):
    # Sort results by BLEU score for plotting
    df_sorted = df.sort_values(by="BLEU Score", ascending=False)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df_sorted['Path'], df_sorted['BLEU Score'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('BLEU Score')
    plt.title(f'BLEU Scores by Transformer Config (EN->{lang.upper()})')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

    if show_plot:
        plt.show()

# === CLI Entry Point ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints",
                        help="Directory with checkpoint subfolders",
                        required=True)
    parser.add_argument('--save_path', type=str, default="checkpoints",
                        help="Subfolder name for saving analysis outputs",
                        required=True)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="List of checkpoint subfolders to exclude")
    parser.add_argument("--show_plot", action="store_true",
                        help="Display plot interactively (optional)")
    parser.add_argument("--lang", required=True,
                        help="Target language: 'fr' or 'de'")
    args = parser.parse_args()

    # Ensure valid language
    if args.lang not in ('fr', 'de'):
        raise ValueError("Language must be in (fr|de)")

    # Load all checkpoint results
    df = load_results(args.checkpoint, args.skip)
    if df.empty:
        print('No valid checkpoints found')
        return

    # Sort and print BLEU score summary
    df_sorted = df.sort_values(by="BLEU Score", ascending=False)
    print("BLEU Scores summary:")
    print(df_sorted.to_string(index=False))

    # Ensure output directory exists
    os.makedirs(os.path.join('analysis', args.save_path), exist_ok=True)

    # Save results to CSV
    csv_path = os.path.join('analysis', args.save_path,
                            f"bleu_scores_summary_en_{args.lang}.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"[Success] Saved CSV to {csv_path}")

    # Save BLEU score plot
    plot_path = os.path.join('analysis', args.save_path,
                             f"bleu_scores_en_{args.lang}.png")
    plot_bleu_scores(df_sorted, lang=args.lang,
                     output_file=plot_path,
                     show_plot=args.show_plot)

# Run if executed directly
if __name__ == '__main__':
    main()