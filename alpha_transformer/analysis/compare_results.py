import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sentencepiece as spm

# Parse model size, d_model, and vocab size from the folder name
def parse_checkpoint_name(name):
    # find regular expression matches
    match = re.match(r"en_(fr|de)_(small|medium|large)_d(\d+)_v(\d+)k", name)
    if match:
        lang, model_size, d_model, vocab_size = match.groups()
        return lang, model_size, int(d_model), int(vocab_size)*1000
    return None, None, None, None

# load model metadata from all of the best.pt models within the selected
# directory
def load_results(folder, skip_list):
    results = []

    for subdir in os.listdir(folder):
        if subdir in skip_list:
            print(f"[Skip] {subdir} is in exclusion list")
            continue

        # get the path to subdirectory
        path = os.path.join(folder, subdir)
        # check that path exists
        if not os.path.isdir(path):
            continue

        # use the best_model.pt
        model_path = os.path.join(path, "best_model.pt")
        if not os.path.exists(model_path):
            print(f"[Missing] Skipping {subdir} - could not find best_model.pt")
            continue

        try:
            # load checkpoint to cpu to avoid GPU usage
            # GPU is being used for training
            checkpoint = torch.load(model_path, map_location="cpu")

            # extract training arguments and key stats
            args = checkpoint.get("args", {})
            lang, model_size, d_model, vocab_size = parse_checkpoint_name(subdir)
            bleu_score = checkpoint.get("best_bleu_score", -1) # get best BLEU score
            epoch = checkpoint.get("epoch", -1)

            # get vocab size
            sp = spm.SentencePieceProcessor()
            sp.load(args['sp_model_path'])
            vocab_size = sp.get_piece_size()

            results.append({
                "Lang":lang,
                "Path": subdir,
                "Model Size": model_size,
                "D_model":d_model or args.get("d_model"),
                'Vocab Size':vocab_size,
                'Epoch': epoch,
                'BLEU Score': bleu_score
            })

        except Exception as e:
            print(f'[Error] Failed to read {subdir}: {e}')

    return pd.DataFrame(results)

# create a bar plot of BLEU score
def plot_bleu_scores(df: pd.DataFrame, lang: str,
                     output_file=None,
                     show_plot=False):
    df_sorted = df.sort_values(by="BLEU Score", ascending=False)

    # make the plot
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
    
# CLI entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints",
                        help="Directory with checkpoint subfolders",
                        required=True)
    parser.add_argument('--save_path', type=str, default="checkpoints",
                        help="Directory with checkpoint subfolders",
                        required=True)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="List of subfolders to skip")
    parser.add_argument("--show_plot", action="store_true",
                        help="Display plot interactively (e.g., in Jupyter)")
    parser.add_argument("--lang", required=True,
                        help='Language in (fr|de)')
    args = parser.parse_args()

    if args.lang not in ('fr', 'de'):
        raise ValueError("Language must be in (fr|de)")

    df = load_results(args.checkpoint, args.skip)
    if df.empty:
        print('No Valid checkpoints found')
        return

    df_sorted = df.sort_values(by="BLEU Score", ascending=False)

    print("BLEU Scores summary:")
    print(df_sorted.to_string(index=False))

    # create args.save_path if it doesn't exist
    os.makedirs(os.path.join('analysis', args.save_path), exist_ok=True)

    csv_path = os.path.join('analysis', args.save_path,
                            f"bleu_scores_summary_en_{args.lang}.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"[Success] Saved CSV to {csv_path}")

    plot_path = os.path.join('analysis', args.save_path,
                             f"bleu_scores_en_{args.lang}.png")
    plot_bleu_scores(df_sorted, lang=args.lang, output_file=plot_path,
                     show_plot=args.show_plot)

if __name__ == '__main__':
    main()