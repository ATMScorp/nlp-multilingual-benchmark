import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
import evaluate
import torch
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')

# Configuration

DATA_DIR = "data"
EUROPARL_EN = os.path.join(DATA_DIR, "europarl", "en-pl.en")
EUROPARL_PL = os.path.join(DATA_DIR, "europarl", "en-pl.pl")
TATOEBA = os.path.join(DATA_DIR, "tatoeba", "en-de.tsv")
TED_EN = os.path.join(DATA_DIR, "ted2020", "en-de.en.txt")
TED_DE = os.path.join(DATA_DIR, "ted2020", "en-de.de.txt")

# Load data

def load_europarl(n=100):
    with open(EUROPARL_EN, encoding='utf-8') as f:
        en = [line.strip() for line in f.readlines()[:n]]
    with open(EUROPARL_PL, encoding='utf-8') as f:
        pl = [line.strip() for line in f.readlines()[:n]]
    return en, pl

def load_tatoeba(path: str, max_pairs: int = 100):
    df = pd.read_csv(path, sep='\t', header=None, names=['EN_ID', 'EN_TEXT', 'DE_ID', 'DE_TEXT'])
    df_unique = df.drop_duplicates(subset='EN_TEXT', keep='first')
    df_sample = df_unique.sample(n=min(max_pairs, len(df_unique)), random_state=42).reset_index(drop=True)
    return df_sample['EN_TEXT'].tolist(), df_sample['DE_TEXT'].tolist()

def load_ted(n=200, max_len=100):
    with open(TED_EN, encoding='utf-8') as f:
        en = [line.strip() for line in f.readlines()]
    with open(TED_DE, encoding='utf-8') as f:
        de = [line.strip() for line in f.readlines()]
    pairs = [(e, d) for e, d in zip(en, de) if len(e.split()) <= max_len and len(d.split()) <= max_len]
    selected = pairs[:n]
    return [e for e, _ in selected], [d for _, d in selected]

# Embedding models

print("\n[INFO] Loading embedding models...")

model_st = SentenceTransformer('distiluse-base-multilingual-cased')
tokenizer_mbart = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model_mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model_mbart.eval()
model_xlm = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Helper functions

def evaluate_all_models_translation(refs, mbart_preds, dataset_name, lang):
    metric_results = []
    scorer = evaluate.load("bertscore")

    def calc_metrics(hyps, refs, name):
        bleu = corpus_bleu([[ref.split()] for ref in refs], [hyp.split() for hyp in hyps])
        chrf = corpus_chrf(refs, hyps)
        bert = scorer.compute(predictions=hyps, references=refs, lang=lang)
        bert_f1 = np.mean(bert["f1"])

        print(f"\n📐 {name} — {dataset_name}")
        print(f"BLEU:      {bleu:.4f}")
        print(f"chrF:      {chrf:.4f}")
        print(f"BERTScore: {bert_f1:.4f}")

        return {
            "Model": name,
            "BLEU": bleu,
            "chrF": chrf,
            "BERTScore": bert_f1
        }

    metric_results.append(calc_metrics(mbart_preds, refs, "mBART Translation"))

    metrics_df = pd.DataFrame(metric_results)
    metrics_df.to_csv(f"results/csv/translation_metrics_{dataset_name}.csv", index=False)

def cache_path(model_name, dataset_name, side):
    os.makedirs("cache", exist_ok=True)
    return f"cache/{dataset_name}_{model_name}_{side}.npy"

def embed_sentences(sentences, model, model_name, dataset_name, side, batch_size=64):
    path = cache_path(model_name, dataset_name, side)
    if os.path.exists(path):
        return np.load(path)

    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Embedding {model_name} {side}"):
        batch = sentences[i:i + batch_size]
        embs = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        embeddings.extend(embs)

    embeddings = np.array(embeddings)
    np.save(path, embeddings)
    return embeddings

def compute_similarities(emb1, emb2):
    return [cosine_similarity([v1], [v2])[0][0] for v1, v2 in zip(emb1, emb2)]

def translate_mbart(texts, src_lang, tgt_lang, batch_size=16):
    tokenizer_mbart.src_lang = src_lang
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        encoded = tokenizer_mbart(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            generated_tokens = model_mbart.generate(
                **encoded,
                forced_bos_token_id=tokenizer_mbart.lang_code_to_id[tgt_lang]
            )
        translations.extend(tokenizer_mbart.batch_decode(generated_tokens, skip_special_tokens=True))
    return translations

# Visualization functions

def plot_similarity_distributions(df, dataset_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Sim_SentenceTransformer"], label="SentenceTransformer", kde=True, stat="density", bins=25)
    sns.histplot(df["Sim_XLMR"], label="XLM-RoBERTa", kde=True, stat="density", bins=25)
    sns.histplot(df["Sim_mBART_Translation"], label="mBART Translation", kde=True, stat="density", bins=25)
    plt.title(f"Similarity Distributions — {dataset_name}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/similarity/similarity_plot_{dataset_name}.png")
    plt.close()


def plot_pca_embeddings(emb_src, emb_tgt, dataset_name):
    emb_all = np.vstack((emb_src, emb_tgt))
    labels = ["src"] * len(emb_src) + ["tgt"] * len(emb_tgt)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(emb_all)

    df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df["Type"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Type", alpha=0.6)
    plt.title(f"PCA — {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"results/plots/pca/pca_{dataset_name}.png")
    plt.close()

def plot_tsne_embeddings(emb_src, emb_tgt, dataset_name, sample_size=100):
    emb_src_sample = emb_src[:sample_size]
    emb_tgt_sample = emb_tgt[:sample_size]
    emb_all = np.vstack((emb_src_sample, emb_tgt_sample))
    labels = ["src"] * len(emb_src_sample) + ["tgt"] * len(emb_tgt_sample)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(emb_all)

    df = pd.DataFrame(reduced, columns=["Dim1", "Dim2"])
    df["Type"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Type", alpha=0.6)
    plt.title(f"t-SNE — {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"results/plots/tsne/tsne_{dataset_name}.png")
    plt.close()

def plot_correlation_heatmap(corr, dataset_name):
    os.makedirs("results/plots/correlation", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap — {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"results/plots/correlation/correlation_heatmap_{dataset_name}.png")
    plt.close()

# Experiment functions

def run_experiment(src_texts, tgt_texts, src_lang, tgt_lang, dataset_name, enable_visuals=True):
    print(f"\n[INFO] Running experiment for {dataset_name} [{src_lang} -> {tgt_lang}]...")

    emb1_st = embed_sentences(src_texts, model_st, "ST", dataset_name, "src")
    emb2_st = embed_sentences(tgt_texts, model_st, "ST", dataset_name, "tgt")
    sim_st = compute_similarities(emb1_st, emb2_st)

    emb1_xlm = embed_sentences(src_texts, model_xlm, "XLMR", dataset_name, "src")
    emb2_xlm = embed_sentences(tgt_texts, model_xlm, "XLMR", dataset_name, "tgt")
    sim_xlm = compute_similarities(emb1_xlm, emb2_xlm)

    translated = translate_mbart(src_texts, src_lang, tgt_lang)
    emb_trans = embed_sentences(translated, model_st, "mBART", dataset_name, "translated")
    sim_trans = compute_similarities(emb_trans, emb2_st)

    df = pd.DataFrame({
        "Sentence_SRC": src_texts,
        "Sentence_TGT": tgt_texts,
        "Sim_SentenceTransformer": sim_st,
        "Sim_XLMR": sim_xlm,
        "Sim_mBART_Translation": sim_trans
    })

    df.to_csv(f"results/csv/results_{dataset_name}.csv", index=False)

    evaluate_all_models_translation(
        refs=tgt_texts,
        mbart_preds=translated,
        dataset_name=dataset_name,
        lang=tgt_lang.split("_")[0]
    )

    if enable_visuals:
        plot_pca_embeddings(emb1_st, emb2_st, f"{dataset_name}_ST")
        plot_tsne_embeddings(emb1_st, emb2_st, f"{dataset_name}_ST")

        plot_pca_embeddings(emb1_xlm, emb2_xlm, f"{dataset_name}_XLMR")
        plot_tsne_embeddings(emb1_xlm, emb2_xlm, f"{dataset_name}_XLMR")

        plot_pca_embeddings(emb_trans, emb2_st, f"{dataset_name}_mBART")
        plot_tsne_embeddings(emb_trans, emb2_st, f"{dataset_name}_mBART")

    plot_similarity_distributions(df, dataset_name)

    corr = df[["Sim_SentenceTransformer", "Sim_XLMR", "Sim_mBART_Translation"]].corr(method='pearson')
    print(f"\n[INFO] Correlation matrix for {dataset_name}:\n{corr}\n")
    corr.to_csv(f"results/csv/correlation_{dataset_name}.csv")
    plot_correlation_heatmap(corr, dataset_name)

    return df


def main():
    euro_en, euro_pl = load_europarl()
    tato_en, tato_de = load_tatoeba(TATOEBA)
    ted_en, ted_de = load_ted()

    df1 = run_experiment(euro_en, euro_pl, "en_XX", "pl_PL", "Europarl")
    df2 = run_experiment(tato_en, tato_de, "en_XX", "de_DE", "Tatoeba")
    df3 = run_experiment(ted_en, ted_de, "en_XX", "de_DE", "TED")

    all_data = [
        ("Europarl", df1),
        ("Tatoeba", df2),
        ("TED", df3)
    ]

    benchmark = []

    for name, df in all_data:
        benchmark.append({
            "Dataset": name,
            "Mean_ST": np.mean(df["Sim_SentenceTransformer"]),
            "Mean_XLMR": np.mean(df["Sim_XLMR"]),
            "Mean_mBART": np.mean(df["Sim_mBART_Translation"]),
            "STD_ST": np.std(df["Sim_SentenceTransformer"]),
            "STD_XLMR": np.std(df["Sim_XLMR"]),
            "STD_mBART": np.std(df["Sim_mBART_Translation"])
        })

    benchmark_df = pd.DataFrame(benchmark)
    benchmark_df.to_csv("results/csv/benchmark_results.csv", index=False)
    print("\n Benchmark table:\n")
    print(benchmark_df)
    print("\n All experiments completed. Results saved to CSV and PNG files.")

if __name__ == "__main__":
    main()
