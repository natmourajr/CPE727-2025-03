"""
Exploratory Data Analysis for AG_NEWS Dataset

Performs comprehensive EDA including:
- Dataset overview
- Raw text analysis
- Text preprocessing exploration
- Feature correlation analysis
- PCA analysis

Outputs are saved to eda/outputs/ag_news/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.covariance import EmpiricalCovariance
import warnings
import plotly.express as px
import plotly.graph_objects as go
import pickle
import argparse

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader_agnews import AGNewsLoader
from src.config import RANDOM_SEED, AG_NEWS_CLASS_NAMES

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "outputs" / "ag_news"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache directory
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
DPI = 300


def load_cached_data(cache_file, use_cache=True):
    """Load data from cache if available and use_cache is True"""
    cache_path = CACHE_DIR / cache_file
    if use_cache and cache_path.exists():
        print(f"  Loading from cache: {cache_file}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


def save_to_cache(data, cache_file):
    """Save data to cache"""
    cache_path = CACHE_DIR / cache_file
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved to cache: {cache_file}")


def save_statistics(stats_dict, filename="statistics_summary"):
    """Save statistics to both TXT and CSV formats"""
    # Save as TXT (human readable)
    txt_path = OUTPUT_DIR / f"{filename}.txt"
    with open(txt_path, 'w') as f:
        for key, value in stats_dict.items():
            f.write(f"{key}: {value}\n")

    # Save as CSV (structured)
    csv_path = OUTPUT_DIR / f"{filename}.csv"
    df = pd.DataFrame([stats_dict])
    df.to_csv(csv_path, index=False)

    print(f"✓ Statistics saved to {txt_path} and {csv_path}")


def plot_dataset_overview(loader, use_cache=True):
    """1. Dataset Overview"""
    print("\n" + "="*60)
    print("1. DATASET OVERVIEW")
    print("="*60)

    # Try to load from cache
    cached = load_cached_data('raw_texts.pkl', use_cache)
    if cached is not None:
        train_texts, train_labels, test_texts, test_labels = cached
    else:
        # Get raw data
        train_texts, train_labels = loader._prepare_data(loader.train_data)
        test_texts, test_labels = loader._prepare_data(loader.test_data)

        # Save to cache
        save_to_cache((train_texts, train_labels, test_texts, test_labels), 'raw_texts.pkl')

    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels

    # Total samples
    stats = {
        "total_samples": len(all_texts),
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "num_classes": 4,
    }

    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Test samples: {stats['test_samples']}")

    # Class distribution (text output only - perfectly balanced dataset)
    print("\nClass distribution:")
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        count = sum(1 for label in train_labels if label == i)
        stats[f"train_{class_name.lower()}_count"] = count
        print(f"  {class_name}: {count} samples")

    return train_texts, train_labels, test_texts, test_labels, stats


def analyze_raw_text(train_texts, train_labels, stats):
    """2. Raw Text Analysis"""
    print("\n" + "="*60)
    print("2. RAW TEXT ANALYSIS")
    print("="*60)

    # Character and word counts
    char_lengths = [len(text) for text in train_texts]
    word_counts = [len(text.split()) for text in train_texts]

    stats["avg_char_length"] = np.mean(char_lengths)
    stats["avg_word_count"] = np.mean(word_counts)
    stats["median_char_length"] = np.median(char_lengths)
    stats["median_word_count"] = np.median(word_counts)

    print(f"\nAverage character length: {stats['avg_char_length']:.1f}")
    print(f"Average word count: {stats['avg_word_count']:.1f}")

    # Per-class analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Character length distribution
    ax = axes[0, 0]
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        class_char_lengths = [len(train_texts[j]) for j in range(len(train_texts)) if train_labels[j] == i]
        ax.hist(class_char_lengths, bins=50, alpha=0.5, label=class_name)
    ax.set_xlabel("Character Length", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Character Length Distribution by Class", fontsize=12, fontweight='bold')
    ax.legend()

    # Word count distribution
    ax = axes[0, 1]
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        class_word_counts = [len(train_texts[j].split()) for j in range(len(train_texts)) if train_labels[j] == i]
        ax.hist(class_word_counts, bins=50, alpha=0.5, label=class_name)
    ax.set_xlabel("Word Count", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Word Count Distribution by Class", fontsize=12, fontweight='bold')
    ax.legend()

    # Box plots
    ax = axes[1, 0]
    char_data = [[len(train_texts[j]) for j in range(len(train_texts)) if train_labels[j] == i]
                 for i in range(4)]
    bp = ax.boxplot(char_data, labels=AG_NEWS_CLASS_NAMES, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", 4)):
        patch.set_facecolor(color)
    ax.set_ylabel("Character Length", fontsize=11)
    ax.set_title("Character Length by Class (Box Plot)", fontsize=12, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax = axes[1, 1]
    word_data = [[len(train_texts[j].split()) for j in range(len(train_texts)) if train_labels[j] == i]
                 for i in range(4)]
    bp = ax.boxplot(word_data, labels=AG_NEWS_CLASS_NAMES, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", 4)):
        patch.set_facecolor(color)
    ax.set_ylabel("Word Count", fontsize=11)
    ax.set_title("Word Count by Class (Box Plot)", fontsize=12, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "text_length_distribution.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: text_length_distribution.png")

    # Sample texts
    print("\nSample texts from each class:")
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        class_indices = [j for j in range(len(train_texts)) if train_labels[j] == i]
        sample_idx = class_indices[0]
        print(f"\n{class_name}: {train_texts[sample_idx][:200]}...")


def analyze_preprocessing(train_texts, train_labels, stats):
    """3. Text Preprocessing Exploration"""
    print("\n" + "="*60)
    print("3. TEXT PREPROCESSING EXPLORATION")
    print("="*60)

    # Subsample for faster preprocessing analysis
    print("\nSubsampling 5,000 documents for preprocessing exploration...")
    n_samples = min(5000, len(train_texts))
    sample_indices = np.random.choice(len(train_texts), n_samples, replace=False)
    texts_sample = [train_texts[i] for i in sample_indices]
    labels_sample = [train_labels[i] for i in sample_indices]

    # a) Vocabulary Analysis (Before Filtering)
    print("\n--- a) Vocabulary Analysis (Before Filtering) ---")

    count_vec_no_filter = CountVectorizer()
    count_vec_no_filter.fit(texts_sample)
    vocab_no_filter = len(count_vec_no_filter.vocabulary_)
    stats["vocab_size_no_filter"] = vocab_no_filter
    print(f"Total unique terms in corpus (5k sample): {vocab_no_filter}")

    # Per-class vocabulary
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        class_texts = [texts_sample[j] for j in range(len(texts_sample)) if labels_sample[j] == i]
        cv = CountVectorizer()
        cv.fit(class_texts)
        print(f"  {class_name}: {len(cv.vocabulary_)} unique terms")

    # b) Stopwords Impact
    print("\n--- b) Stopwords Impact ---")

    count_vec_with_stopwords = CountVectorizer(stop_words=None)
    count_vec_with_stopwords.fit(texts_sample)
    vocab_with_stopwords = len(count_vec_with_stopwords.vocabulary_)

    count_vec_no_stopwords = CountVectorizer(stop_words='english')
    count_vec_no_stopwords.fit(texts_sample)
    vocab_no_stopwords = len(count_vec_no_stopwords.vocabulary_)

    reduction = (vocab_with_stopwords - vocab_no_stopwords) / vocab_with_stopwords * 100
    stats["vocab_before_stopwords"] = vocab_with_stopwords
    stats["vocab_after_stopwords"] = vocab_no_stopwords
    stats["stopwords_reduction_pct"] = reduction

    print(f"Vocabulary before stopwords removal: {vocab_with_stopwords}")
    print(f"Vocabulary after stopwords removal: {vocab_no_stopwords}")
    print(f"Reduction: {reduction:.1f}%")

    # c) Frequency Filtering Impact
    print("\n--- c) Frequency Filtering Impact ---")

    # max_df filtering
    tfidf_no_maxdf = TfidfVectorizer(stop_words='english', min_df=5)
    tfidf_no_maxdf.fit(texts_sample)
    vocab_no_maxdf = len(tfidf_no_maxdf.vocabulary_)

    tfidf_with_maxdf = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.5)
    tfidf_with_maxdf.fit(texts_sample)
    vocab_with_maxdf = len(tfidf_with_maxdf.vocabulary_)

    removed_by_maxdf = vocab_no_maxdf - vocab_with_maxdf
    print(f"Terms removed by max_df=0.5: {removed_by_maxdf}")

    # min_df filtering
    tfidf_no_mindf = TfidfVectorizer(stop_words='english', max_df=0.5)
    tfidf_no_mindf.fit(texts_sample)
    vocab_no_mindf = len(tfidf_no_mindf.vocabulary_)

    removed_by_mindf = vocab_no_mindf - vocab_with_maxdf
    print(f"Terms removed by min_df=5: {removed_by_mindf}")

    # Final vocabulary (with max_features=10000) - FIT ON FULL DATASET
    print("\n--- Fitting final vectorizer on FULL dataset ---")
    tfidf_final = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5, max_features=10000, sublinear_tf=True)
    tfidf_final.fit(train_texts)
    vocab_final = len(tfidf_final.vocabulary_)
    stats["vocab_final"] = vocab_final
    print(f"Final vocabulary size (max_features=10000): {vocab_final}")

    # Visualization of preprocessing impact
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = ['Raw\nVocabulary\n(5k sample)', 'After\nStopwords\n(5k sample)', 'After\nmin_df=5\nmax_df=0.5\n(5k sample)', 'Final\n(max_features=10k)\n(120k full dataset)']
    sizes = [vocab_no_filter, vocab_no_stopwords, vocab_with_maxdf, vocab_final]

    # Use different colors for sample vs full dataset
    colors = ['#6baed6', '#6baed6', '#6baed6', '#08519c']  # lighter blue for sample, darker for full
    bars = ax.bar(steps, sizes, color=colors)

    ax.set_ylabel("Vocabulary Size", fontsize=12)
    ax.set_title("Impact of Preprocessing Steps on Vocabulary Size", fontsize=14, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add explanation note
    note_text = ("Note: First 3 steps use 5k sample for exploration speed.\n"
                 "Final step uses full 120k training dataset for actual model.\n"
                 "More terms survive min_df=5 on larger dataset (120k vs 5k).")
    ax.text(0.5, 0.02, note_text, transform=ax.transAxes,
            fontsize=9, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "preprocessing_impact.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: preprocessing_impact.png")

    return tfidf_final


def analyze_vocabulary_patterns(train_texts, stats, use_cache=True):
    """3d. Vocabulary and Language Patterns"""
    print("\n--- d) Vocabulary and Language Patterns ---")

    # Try to load from cache
    cached = load_cached_data('tokenization_results.pkl', use_cache)
    if cached is not None:
        token_freq, total_tokens, unique_tokens = cached
        print(f"Total tokens in corpus: {total_tokens:,}")
        print(f"Unique tokens (vocabulary size): {unique_tokens:,}")
    else:
        # Subsample for analysis (use all training data but analyze efficiently)
        print("Building token frequency distribution...")

        # Simple tokenization (split by whitespace and punctuation)
        from collections import Counter
        import re

        all_tokens = []
        for text in train_texts:
            # Simple tokenization: lowercase and split
            tokens = re.findall(r'\b\w+\b', text.lower())
            all_tokens.extend(tokens)

        # Build frequency distribution
        token_freq = Counter(all_tokens)
        total_tokens = len(all_tokens)
        unique_tokens = len(token_freq)

        # Save to cache
        save_to_cache((token_freq, total_tokens, unique_tokens), 'tokenization_results.pkl')
        print(f"Total tokens in corpus: {total_tokens:,}")
        print(f"Unique tokens (vocabulary size): {unique_tokens:,}")

    stats['total_tokens'] = total_tokens
    stats['unique_tokens'] = unique_tokens
    stats['vocabulary_size_raw'] = unique_tokens
    print(f"Type-token ratio: {unique_tokens/total_tokens:.4f}")

    # Sort by frequency
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)

    # Calculate vocabulary coverage
    cumulative_count = 0
    coverage_thresholds = [0.80, 0.90, 0.95, 0.99]
    coverage_results = {}

    for threshold in coverage_thresholds:
        target = total_tokens * threshold
        for idx, (token, count) in enumerate(sorted_tokens):
            cumulative_count += count
            if cumulative_count >= target:
                coverage_results[threshold] = idx + 1
                stats[f'tokens_for_{int(threshold*100)}pct_coverage'] = idx + 1
                print(f"Tokens needed for {int(threshold*100)}% coverage: {idx + 1:,} ({idx+1/unique_tokens*100:.2f}% of vocabulary)")
                break
        cumulative_count = 0  # Reset for next threshold

    # Analyze rare tokens
    singleton_count = sum(1 for _, count in token_freq.items() if count == 1)
    rare_count = sum(1 for _, count in token_freq.items() if count <= 5)

    stats['singleton_tokens'] = singleton_count
    stats['rare_tokens_le5'] = rare_count
    stats['singleton_pct'] = singleton_count / unique_tokens * 100
    stats['rare_tokens_pct'] = rare_count / unique_tokens * 100

    print(f"\nRare token analysis:")
    print(f"  Singletons (count=1): {singleton_count:,} ({singleton_count/unique_tokens*100:.2f}% of vocabulary)")
    print(f"  Rare tokens (count≤5): {rare_count:,} ({rare_count/unique_tokens*100:.2f}% of vocabulary)")

    # Top 20 most frequent tokens
    print(f"\nTop 20 most frequent tokens:")
    for token, count in sorted_tokens[:20]:
        print(f"  {token}: {count:,} ({count/total_tokens*100:.3f}%)")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Zipf's law: Rank vs Frequency (log-log scale)
    ax = axes[0, 0]
    ranks = np.arange(1, len(sorted_tokens) + 1)
    frequencies = [count for _, count in sorted_tokens]
    ax.loglog(ranks, frequencies, linewidth=2, alpha=0.7)
    ax.set_xlabel("Rank (log scale)", fontsize=11)
    ax.set_ylabel("Frequency (log scale)", fontsize=11)
    ax.set_title("Zipf's Law: Token Rank vs Frequency", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Cumulative coverage
    ax = axes[0, 1]
    cumulative_coverage = np.cumsum(frequencies) / total_tokens * 100
    ax.plot(ranks, cumulative_coverage, linewidth=2)
    ax.axhline(80, color='red', linestyle='--', alpha=0.7, label='80%')
    ax.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90%')
    ax.axhline(95, color='green', linestyle='--', alpha=0.7, label='95%')
    ax.axhline(99, color='blue', linestyle='--', alpha=0.7, label='99%')
    ax.set_xlabel("Number of Unique Tokens", fontsize=11)
    ax.set_ylabel("Cumulative Coverage (%)", fontsize=11)
    ax.set_title("Vocabulary Coverage", fontsize=12, fontweight='bold')
    ax.set_xlim(0, min(50000, len(sorted_tokens)))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Frequency distribution histogram
    ax = axes[1, 0]
    freq_counts = [count for _, count in token_freq.items()]
    ax.hist(freq_counts, bins=100, edgecolor='black', alpha=0.7, range=(1, 100))
    ax.set_xlabel("Token Frequency", fontsize=11)
    ax.set_ylabel("Number of Tokens", fontsize=11)
    ax.set_title("Token Frequency Distribution (freq ≤ 100)", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 4. Coverage bar chart
    ax = axes[1, 1]
    threshold_labels = [f'{int(t*100)}%' for t in coverage_thresholds]
    tokens_needed = [coverage_results[t] for t in coverage_thresholds]
    bars = ax.bar(threshold_labels, tokens_needed, color=sns.color_palette("viridis", 4))
    ax.set_ylabel("Number of Unique Tokens", fontsize=11)
    ax.set_xlabel("Coverage Threshold", fontsize=11)
    ax.set_title("Tokens Needed for Coverage Thresholds", fontsize=12, fontweight='bold')

    for bar, count in zip(bars, tokens_needed):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vocabulary_patterns.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: vocabulary_patterns.png")


def analyze_tfidf(train_texts, train_labels, vectorizer, stats):
    """e) TF-IDF Characteristics"""
    print("\n--- e) TF-IDF Characteristics ---")

    # Transform texts (use 10k sample for analysis)
    print("Transforming 10,000 samples for TF-IDF analysis...")
    n_samples = min(10000, len(train_texts))
    sample_indices = np.random.choice(len(train_texts), n_samples, replace=False)
    texts_sample = [train_texts[i] for i in sample_indices]
    labels_sample = [train_labels[i] for i in sample_indices]

    X_tfidf = vectorizer.transform(texts_sample)

    # Sparsity
    sparsity = (X_tfidf.data.size / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100
    stats["tfidf_sparsity_pct"] = 100 - sparsity
    stats["tfidf_nonzero_pct"] = sparsity
    print(f"TF-IDF matrix sparsity: {100 - sparsity:.2f}% zeros")

    # Distribution of TF-IDF values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(X_tfidf.data, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("TF-IDF Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Non-Zero TF-IDF Values", fontsize=12, fontweight='bold')

    # Sparsity visualization
    ax = axes[1]
    categories = ['Non-zero\nValues', 'Zero\nValues']
    values = [sparsity, 100 - sparsity]
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax.set_title("TF-IDF Matrix Sparsity", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tfidf_sparsity.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tfidf_sparsity.png")

    # Top terms per class (work with sparse matrix to avoid memory issues)
    print("\nTop 20 terms per class by average TF-IDF score:")
    feature_names = vectorizer.get_feature_names_out()

    top_terms_by_class = {}
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        class_mask = np.array(labels_sample) == i
        # Use sparse mean calculation
        X_class = X_tfidf[class_mask]
        class_tfidf_mean = np.asarray(X_class.mean(axis=0)).flatten()
        top_indices = class_tfidf_mean.argsort()[-20:][::-1]
        top_terms = [(feature_names[idx], class_tfidf_mean[idx]) for idx in top_indices]
        top_terms_by_class[class_name] = top_terms

        print(f"\n{class_name}:")
        for term, score in top_terms[:10]:
            print(f"  {term}: {score:.4f}")

    # Heatmap of top terms across classes
    # Get top 15 terms for each class
    all_top_terms = set()
    for terms in top_terms_by_class.values():
        all_top_terms.update([term for term, _ in terms[:15]])

    # Create heatmap data
    heatmap_data = []
    term_list = list(all_top_terms)
    for term in term_list:
        row = []
        for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
            class_mask = np.array(labels_sample) == i
            term_idx = np.where(feature_names == term)[0]
            if len(term_idx) > 0:
                X_class = X_tfidf[class_mask]
                row.append(X_class[:, term_idx[0]].mean())
            else:
                row.append(0)
        heatmap_data.append(row)

    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(heatmap_data, xticklabels=AG_NEWS_CLASS_NAMES, yticklabels=term_list,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Average TF-IDF'},
                annot=True, fmt='.3f', annot_kws={'size': 7})
    ax.set_title("Top Terms by Class (Average TF-IDF)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_terms_heatmap.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: top_terms_heatmap.png")

    return X_tfidf


def analyze_class_specific_patterns(train_texts, train_labels, vectorizer, stats):
    """f) Class-Specific Patterns Analysis"""
    print("\n--- f) Class-Specific Patterns Analysis ---")

    # Use subsample for efficiency
    print("Analyzing class-specific patterns on 20,000 samples...")
    n_samples = min(20000, len(train_texts))
    sample_indices = np.random.choice(len(train_texts), n_samples, replace=False)
    texts_sample = [train_texts[i] for i in sample_indices]
    labels_sample = [train_labels[i] for i in sample_indices]

    # Get TF-IDF representations
    X_tfidf = vectorizer.transform(texts_sample)
    feature_names = vectorizer.get_feature_names_out()

    # For each class, compute:
    # 1. Word frequency distributions
    # 2. Log-odds ratios
    # 3. Most discriminative terms

    from scipy.sparse import vstack
    from scipy.stats import chi2_contingency

    class_term_stats = {}

    print("\nComputing class-specific term statistics...")

    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        class_mask = np.array(labels_sample) == i
        other_mask = ~class_mask

        X_class = X_tfidf[class_mask]
        X_other = X_tfidf[other_mask]

        # Mean TF-IDF per term in this class vs others
        class_mean = np.asarray(X_class.mean(axis=0)).flatten()
        other_mean = np.asarray(X_other.mean(axis=0)).flatten()

        # Log-odds ratio (with smoothing)
        epsilon = 1e-10
        log_odds = np.log((class_mean + epsilon) / (other_mean + epsilon))

        # Get top discriminative terms by log-odds
        top_indices = log_odds.argsort()[-30:][::-1]

        class_term_stats[class_name] = {
            'top_terms': [(feature_names[idx], log_odds[idx], class_mean[idx]) for idx in top_indices],
            'class_mean': class_mean,
            'other_mean': other_mean,
            'log_odds': log_odds
        }

    # Print top discriminative terms per class
    print("\nTop 15 discriminative terms per class (by log-odds ratio):")
    for class_name in AG_NEWS_CLASS_NAMES:
        print(f"\n{class_name}:")
        for term, log_odds_val, tfidf_val in class_term_stats[class_name]['top_terms'][:15]:
            print(f"  {term}: log-odds={log_odds_val:.3f}, tfidf={tfidf_val:.4f}")

    # Visualization: Heatmap of log-odds ratios for top terms
    print("\nGenerating class-specific patterns visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Log-odds heatmap
    ax = axes[0, 0]
    # Get top 20 terms for each class
    all_top_terms = []
    for class_name in AG_NEWS_CLASS_NAMES:
        all_top_terms.extend([term for term, _, _ in class_term_stats[class_name]['top_terms'][:20]])
    all_top_terms = list(dict.fromkeys(all_top_terms))  # Remove duplicates, preserve order

    # Create log-odds matrix
    logodds_matrix = []
    for term in all_top_terms:
        row = []
        term_idx = np.where(feature_names == term)[0]
        if len(term_idx) > 0:
            for class_name in AG_NEWS_CLASS_NAMES:
                row.append(class_term_stats[class_name]['log_odds'][term_idx[0]])
        logodds_matrix.append(row)

    sns.heatmap(logodds_matrix, xticklabels=AG_NEWS_CLASS_NAMES, yticklabels=all_top_terms,
                cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': 'Log-Odds Ratio'})
    ax.set_title("Discriminative Terms by Class (Log-Odds Ratio)", fontsize=12, fontweight='bold')

    # 2. Class vs Others comparison for first class
    ax = axes[0, 1]
    class_name = AG_NEWS_CLASS_NAMES[0]
    top_terms = [term for term, _, _ in class_term_stats[class_name]['top_terms'][:15]]
    class_means = [class_term_stats[class_name]['class_mean'][np.where(feature_names == term)[0][0]]
                   for term in top_terms]
    other_means = [class_term_stats[class_name]['other_mean'][np.where(feature_names == term)[0][0]]
                   for term in top_terms]

    x = np.arange(len(top_terms))
    width = 0.35
    ax.barh(x - width/2, class_means, width, label=f'{class_name}', alpha=0.8)
    ax.barh(x + width/2, other_means, width, label='Others', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(top_terms, fontsize=8)
    ax.set_xlabel('Mean TF-IDF', fontsize=10)
    ax.set_title(f'{class_name}: Class vs Others', fontsize=12, fontweight='bold')
    ax.legend()
    ax.invert_yaxis()

    # 3. Log-odds distribution across all terms
    ax = axes[1, 0]
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        log_odds_vals = class_term_stats[class_name]['log_odds']
        # Remove inf and -inf
        log_odds_vals = log_odds_vals[np.isfinite(log_odds_vals)]
        ax.hist(log_odds_vals, bins=50, alpha=0.5, label=class_name)
    ax.set_xlabel('Log-Odds Ratio', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Distribution of Log-Odds Ratios', fontsize=12, fontweight='bold')
    ax.legend()
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)

    # 4. Top terms comparison across classes
    ax = axes[1, 1]
    # For each class, show its top 5 unique discriminative terms
    unique_terms_per_class = {}
    for class_name in AG_NEWS_CLASS_NAMES:
        # Get terms that are highly discriminative for this class
        class_tops = [term for term, log_odds_val, _ in class_term_stats[class_name]['top_terms'][:10]
                      if log_odds_val > 0.5]  # Only strongly discriminative
        unique_terms_per_class[class_name] = class_tops[:5]

    # Create a grouped bar chart
    all_unique_terms = []
    for terms in unique_terms_per_class.values():
        all_unique_terms.extend(terms)
    all_unique_terms = list(dict.fromkeys(all_unique_terms))[:20]  # Limit to 20 terms

    term_class_matrix = []
    for term in all_unique_terms:
        row = []
        term_idx = np.where(feature_names == term)[0]
        if len(term_idx) > 0:
            for class_name in AG_NEWS_CLASS_NAMES:
                row.append(class_term_stats[class_name]['class_mean'][term_idx[0]])
        term_class_matrix.append(row)

    x = np.arange(len(all_unique_terms))
    width = 0.2
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        values = [row[i] for row in term_class_matrix]
        ax.bar(x + i*width, values, width, label=class_name, alpha=0.8)

    ax.set_ylabel('Mean TF-IDF', fontsize=10)
    ax.set_title('Top Discriminative Terms Across Classes', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(all_unique_terms, rotation=45, ha='right', fontsize=8)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_specific_patterns.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: class_specific_patterns.png")

    # NEW: Class vs Others comparison for ALL classes (2x2 grid)
    print("Generating class vs others comparison for all classes...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        ax = axes[i // 2, i % 2]

        # Get top 15 discriminative terms for this class
        top_terms = [term for term, _, _ in class_term_stats[class_name]['top_terms'][:15]]
        class_means = [class_term_stats[class_name]['class_mean'][np.where(feature_names == term)[0][0]]
                       for term in top_terms]
        other_means = [class_term_stats[class_name]['other_mean'][np.where(feature_names == term)[0][0]]
                       for term in top_terms]

        # Create horizontal bar chart
        x = np.arange(len(top_terms))
        width = 0.35
        ax.barh(x - width/2, class_means, width, label=f'{class_name}', alpha=0.8)
        ax.barh(x + width/2, other_means, width, label='Others', alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(top_terms, fontsize=8)
        ax.set_xlabel('Mean TF-IDF', fontsize=10)
        ax.set_title(f'{class_name}: Class vs Others', fontsize=12, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_vs_others_all.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: class_vs_others_all.png")

    # NEW: Word frequency distribution across documents for each class
    print("Generating word frequency distribution across documents for each class...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    from collections import Counter, defaultdict
    import re

    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        ax = axes[i // 2, i % 2]

        # Get texts for this class
        class_texts = [texts_sample[j] for j in range(len(texts_sample)) if labels_sample[j] == i]

        # Count word frequencies across all documents
        total_word_counts = Counter()
        for text in class_texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            total_word_counts.update(tokens)

        # Get top 15 most frequent words
        top_words = [word for word, _ in total_word_counts.most_common(15)]

        # For each word, track counts per document
        word_doc_counts = defaultdict(list)
        for word in top_words:
            for doc_idx, text in enumerate(class_texts):
                tokens = re.findall(r'\b\w+\b', text.lower())
                count = tokens.count(word)
                if count > 0:
                    word_doc_counts[word].append(count)

        # Create stacked bar chart
        y_pos = np.arange(len(top_words))

        # Generate colors for documents (use colormap for variety)
        cmap = plt.cm.get_cmap('tab20c')

        for word_idx, word in enumerate(top_words):
            doc_counts = word_doc_counts[word]
            left = 0

            # Sort counts to make visualization cleaner
            doc_counts_sorted = sorted(doc_counts, reverse=True)

            # Create stacked segments
            for seg_idx, count in enumerate(doc_counts_sorted):
                color = cmap(seg_idx % 20)
                ax.barh(word_idx, count, left=left, height=0.8,
                       color=color, edgecolor='white', linewidth=0.5)
                left += count

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words, fontsize=9)
        ax.set_xlabel('Total Word Count (stacked by document)', fontsize=10)
        ax.set_title(f'{class_name}: Word Distribution Across Documents',
                    fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "word_frequency_distribution_per_class.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: word_frequency_distribution_per_class.png")

    # Save top discriminative terms to file
    with open(OUTPUT_DIR / "discriminative_terms.txt", 'w') as f:
        f.write("Top Discriminative Terms Per Class (Log-Odds Ratio)\n")
        f.write("="*60 + "\n\n")
        for class_name in AG_NEWS_CLASS_NAMES:
            f.write(f"{class_name}:\n")
            for term, log_odds_val, tfidf_val in class_term_stats[class_name]['top_terms'][:30]:
                f.write(f"  {term}: log-odds={log_odds_val:.3f}, tfidf={tfidf_val:.4f}\n")
            f.write("\n")

        # Add methodology explanation
        f.write("\n" + "="*60 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*60 + "\n\n")
        f.write("Discriminative terms are identified using log-odds ratios:\n\n")
        f.write("1. For each class C, documents are split into two groups:\n")
        f.write("   - Group 1: Documents belonging to class C\n")
        f.write("   - Group 2: All other documents (not in class C)\n\n")
        f.write("2. For each term T, we compute:\n")
        f.write("   - mean_C: Average TF-IDF score of term T in class C documents\n")
        f.write("   - mean_other: Average TF-IDF score of term T in other documents\n\n")
        f.write("3. Log-odds ratio = log((mean_C + ε) / (mean_other + ε))\n")
        f.write("   where ε = 1e-10 (smoothing to avoid division by zero)\n\n")
        f.write("4. HIGH positive log-odds → term is MUCH more common in class C\n")
        f.write("   → DISCRIMINATIVE for that class\n\n")
        f.write("5. Terms are ranked by log-odds (highest first)\n\n")
        f.write("Sample size: 20,000 documents\n")
        f.write("TF-IDF vectorizer settings:\n")
        f.write("  - stop_words='english'\n")
        f.write("  - max_df=0.5 (removes terms in >50% of documents)\n")
        f.write("  - min_df=5 (removes terms in <5 documents)\n")
        f.write("  - max_features=10000\n")
        f.write("  - sublinear_tf=True\n")
    print("✓ Saved: discriminative_terms.txt")

    # Save less discriminative terms (shared vocabulary across classes)
    print("Identifying less discriminative terms (shared vocabulary)...")

    # Collect all log-odds values across all classes for each term
    term_logodds_variance = {}
    for term_idx, term in enumerate(feature_names):
        logodds_values = []
        for class_name in AG_NEWS_CLASS_NAMES:
            logodds_values.append(class_term_stats[class_name]['log_odds'][term_idx])

        # Calculate how balanced this term is across classes
        # Low variance = term appears similarly across classes
        # Also check if log-odds are close to 0
        abs_mean_logodds = np.abs(np.mean(logodds_values))
        variance_logodds = np.var(logodds_values)

        # Score: combine low absolute mean and low variance
        # Lower score = more balanced/less discriminative
        balance_score = abs_mean_logodds + variance_logodds

        term_logodds_variance[term] = {
            'balance_score': balance_score,
            'mean_logodds': np.mean(logodds_values),
            'variance': variance_logodds,
            'class_means': {class_name: class_term_stats[class_name]['class_mean'][term_idx]
                           for class_name in AG_NEWS_CLASS_NAMES}
        }

    # Sort by balance score (lowest = most balanced = least discriminative)
    sorted_balanced_terms = sorted(term_logodds_variance.items(),
                                  key=lambda x: x[1]['balance_score'])

    with open(OUTPUT_DIR / "less_discriminative_terms.txt", 'w') as f:
        f.write("Less Discriminative Terms (Shared Vocabulary Across Classes)\n")
        f.write("="*60 + "\n\n")
        f.write("Top 100 terms that appear most equally across all classes:\n\n")

        for term, stats_dict in sorted_balanced_terms[:100]:
            f.write(f"{term}:\n")
            f.write(f"  Balance score: {stats_dict['balance_score']:.4f} (lower = more balanced)\n")
            f.write(f"  Mean log-odds: {stats_dict['mean_logodds']:.4f} (closer to 0 = more balanced)\n")
            f.write(f"  Variance: {stats_dict['variance']:.4f} (lower = more consistent across classes)\n")
            f.write(f"  Mean TF-IDF per class:\n")
            for class_name in AG_NEWS_CLASS_NAMES:
                f.write(f"    {class_name}: {stats_dict['class_means'][class_name]:.4f}\n")
            f.write("\n")

        # Add methodology explanation
        f.write("\n" + "="*60 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*60 + "\n\n")
        f.write("Less discriminative terms are identified as the 'shared vocabulary'\n")
        f.write("that appears with similar frequency across ALL classes.\n\n")
        f.write("Key differences from discriminative terms:\n")
        f.write("- Discriminative terms: High in one class, low in others\n")
        f.write("- Less discriminative terms: Similar presence across all classes\n\n")
        f.write("These terms have already SURVIVED preprocessing filters:\n")
        f.write("  ✓ NOT stopwords (e.g., 'the', 'a', 'is')\n")
        f.write("  ✓ NOT too rare (appear in ≥5 documents)\n")
        f.write("  ✓ NOT too common (appear in ≤50% of documents)\n")
        f.write("  ✓ Part of final vocabulary (top 10,000 features)\n\n")
        f.write("So these are MEANINGFUL terms that are simply shared across\n")
        f.write("categories rather than class-specific.\n\n")
        f.write("Identification method:\n\n")
        f.write("1. For each term T, collect log-odds ratios from all 4 classes\n\n")
        f.write("2. Compute balance metrics:\n")
        f.write("   - mean_logodds: Average log-odds across classes\n")
        f.write("   - variance: How much log-odds varies across classes\n\n")
        f.write("3. Balance score = |mean_logodds| + variance\n")
        f.write("   - LOW score → term appears EQUALLY across classes\n")
        f.write("   - HIGH score → term is BIASED toward specific classes\n\n")
        f.write("4. Terms ranked by balance score (lowest = most shared)\n\n")
        f.write("Interpretation:\n")
        f.write("- These terms represent common news vocabulary (e.g., 'said',\n")
        f.write("  'year', 'new', 'time') that don't help distinguish between\n")
        f.write("  World, Sports, Business, and Sci/Tech news.\n")
        f.write("- Useful for understanding domain vocabulary vs. class-specific\n")
        f.write("  terminology.\n\n")
        f.write("Sample size: 20,000 documents\n")
        f.write("TF-IDF vectorizer settings: (same as discriminative terms)\n")
        f.write("  - stop_words='english'\n")
        f.write("  - max_df=0.5, min_df=5, max_features=10000, sublinear_tf=True\n")

    print("✓ Saved: less_discriminative_terms.txt")


def analyze_title_vs_description(train_texts, train_labels):
    """g) Title vs Description Analysis"""
    print("\n--- g) Title vs Description Analysis ---")
    print("Note: AG_NEWS format is 'title: description', analyzing as combined text.")
    print("(Skipping detailed title/description split as format may vary)")
    # This analysis would require parsing the specific format of AG_NEWS
    # which combines title and description. Skipping detailed implementation
    # as the exact format is not always consistent.


def analyze_feature_correlation(X_tfidf, stats):
    """4. Feature Correlation Analysis"""
    print("\n" + "="*60)
    print("4. FEATURE CORRELATION ANALYSIS")
    print("="*60)

    # Sample subset for correlation (computing 10k x 10k is expensive)
    # Also subsample documents to speed up computation
    print("\nComputing correlation on random 100-feature subset and 5000 documents...")
    n_features_sample = min(100, X_tfidf.shape[1])
    n_docs_sample = min(5000, X_tfidf.shape[0])

    random_features = np.random.choice(X_tfidf.shape[1], n_features_sample, replace=False)
    random_docs = np.random.choice(X_tfidf.shape[0], n_docs_sample, replace=False)

    X_sample = X_tfidf[random_docs, :][:, random_features].toarray()

    # Correlation matrix
    corr_matrix = np.corrcoef(X_sample.T)

    # Remove diagonal (self-correlation)
    corr_matrix_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_matrix_no_diag, 0)

    # Statistics
    mean_abs_corr = np.abs(corr_matrix_no_diag).mean()
    stats["mean_abs_correlation"] = mean_abs_corr

    print(f"Mean absolute correlation: {mean_abs_corr:.4f}")

    # Distribution of correlations
    corr_values = corr_matrix_no_diag[np.triu_indices_from(corr_matrix_no_diag, k=1)]

    thresholds = [0.1, 0.3, 0.5]
    for thresh in thresholds:
        pct = (np.abs(corr_values) > thresh).sum() / len(corr_values) * 100
        stats[f"corr_above_{thresh}"] = pct
        print(f"Percentage of |correlation| > {thresh}: {pct:.2f}%")

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Correlation matrix heatmap
    ax = axes[0]
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                ax=ax, cbar_kws={'label': 'Correlation'}, square=True)
    ax.set_title(f"Feature Correlation Matrix\n(Random {n_features_sample} features)",
                 fontsize=12, fontweight='bold')

    # Correlation distribution
    ax = axes[1]
    ax.hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(mean_abs_corr, color='red', linestyle='--', linewidth=2, label=f'Mean |corr|={mean_abs_corr:.3f}')
    ax.axvline(-mean_abs_corr, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Correlation Coefficient", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Feature Correlations", fontsize=12, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_analysis.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: correlation_analysis.png")


def analyze_pca(X_tfidf, train_labels, stats):
    """5. PCA Analysis"""
    print("\n" + "="*60)
    print("5. PCA ANALYSIS")
    print("="*60)

    # Subsample for PCA (use 10k samples for faster computation)
    print("\nSubsampling 10,000 documents for PCA analysis...")
    n_samples = min(10000, X_tfidf.shape[0])
    sample_indices = np.random.choice(X_tfidf.shape[0], n_samples, replace=False)
    X_sample = X_tfidf[sample_indices].toarray()
    labels_sample = [train_labels[i] for i in sample_indices]

    # Fit PCA
    print("Fitting PCA...")
    pca = PCA(random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_sample)

    # Cumulative variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    # Find number of components for thresholds
    for threshold in [90, 95, 99]:
        n_components = np.argmax(cumvar >= threshold) + 1
        stats[f"pca_components_for_{threshold}pct_var"] = n_components
        print(f"Components for {threshold}% variance: {n_components}")

    # a) Cumulative variance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=2)
    ax.axhline(90, color='red', linestyle='--', alpha=0.7, label='90%')
    ax.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax.axhline(99, color='green', linestyle='--', alpha=0.7, label='99%')
    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
    ax.set_title("PCA: Cumulative Variance Explained", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_cumulative_variance.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_cumulative_variance.png")

    # b) Scree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    variance_pct = pca.explained_variance_ratio_[:50] * 100  # First 50 components
    ax.bar(range(1, len(variance_pct) + 1), variance_pct)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    ax.set_title("PCA Scree Plot (First 50 Components)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_scree_plot.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_scree_plot.png")

    # c) 2D projection
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", 4)
    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        mask = np.array(labels_sample) == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]],
                  label=class_name, alpha=0.6, s=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
    ax.set_title("PCA: 2D Projection (First 2 Components)", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_2d_projection.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_2d_projection.png")

    # d) Interactive 3D projection
    print("\nCreating interactive 3D plot...")
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Class': [AG_NEWS_CLASS_NAMES[label] for label in labels_sample]
    })

    fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Class',
                       title='PCA: Interactive 3D Projection',
                       labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)',
                              'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)',
                              'PC3': f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)'},
                       opacity=0.6)
    fig.write_html(str(OUTPUT_DIR / "pca_3d_projection.html"))
    print("✓ Saved: pca_3d_projection.html (interactive 3D)")

    # e) Covariance comparison (diagonal vs full)
    print("\n--- PCA-based Covariance Comparison ---")
    print("Comparing diagonal vs full covariance on 2D PCA projection...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for i, class_name in enumerate(AG_NEWS_CLASS_NAMES):
        ax = axes[i // 2, i % 2]
        mask = np.array(labels_sample) == i
        X_class = X_pca[mask, :2]

        # Plot data points
        ax.scatter(X_class[:, 0], X_class[:, 1], alpha=0.3, s=10, label='Data')

        # Fit diagonal covariance Gaussian
        mean = X_class.mean(axis=0)
        cov_diag = np.diag(X_class.var(axis=0))

        # Fit full covariance Gaussian
        cov_full = np.cov(X_class.T)

        # Plot ellipses
        from matplotlib.patches import Ellipse

        # Diagonal covariance ellipse
        eigenvalues_diag = np.diag(cov_diag)
        angle_diag = 0
        width_diag, height_diag = 2 * np.sqrt(eigenvalues_diag) * 2
        ellipse_diag = Ellipse(mean, width_diag, height_diag, angle=angle_diag,
                              facecolor='none', edgecolor='red', linewidth=2,
                              linestyle='--', label='Diagonal Cov')
        ax.add_patch(ellipse_diag)

        # Full covariance ellipse
        eigenvalues_full, eigenvectors_full = np.linalg.eig(cov_full)
        angle_full = np.degrees(np.arctan2(eigenvectors_full[1, 0], eigenvectors_full[0, 0]))
        width_full, height_full = 2 * np.sqrt(eigenvalues_full) * 2
        ellipse_full = Ellipse(mean, width_full, height_full, angle=angle_full,
                              facecolor='none', edgecolor='blue', linewidth=2,
                              label='Full Cov')
        ax.add_patch(ellipse_full)

        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.set_title(f"{class_name}: Diagonal vs Full Covariance", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_covariance_comparison.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_covariance_comparison.png")

    return pca


def generate_profiling_report(train_texts, train_labels):
    """Generate comprehensive HTML report using ydata-profiling"""
    print("\n" + "="*60)
    print("6. GENERATING COMPREHENSIVE HTML REPORTS (ONE PER CLASS)")
    print("="*60)

    try:
        from ydata_profiling import ProfileReport

        # Generate one report per class
        for class_idx, class_name in enumerate(AG_NEWS_CLASS_NAMES):
            print(f"\nGenerating report for {class_name}...")

            # Get indices for this class
            class_indices = [i for i in range(len(train_texts)) if train_labels[i] == class_idx]

            # Subsample for faster profiling (500 samples per class)
            n_samples = min(500, len(class_indices))
            sample_indices = np.random.choice(class_indices, n_samples, replace=False)

            # Create DataFrame with text content for word cloud
            df = pd.DataFrame({
                'text': [train_texts[i] for i in sample_indices],
                'char_length': [len(train_texts[i]) for i in sample_indices],
                'word_count': [len(train_texts[i].split()) for i in sample_indices]
            })

            print(f"  Creating profiling report for {class_name} ({n_samples} samples)...")
            profile = ProfileReport(df,
                                   title=f"AG_NEWS Dataset - {class_name} Class Report",
                                   explorative=True,
                                   minimal=True)

            output_path = OUTPUT_DIR / f"eda_report_{class_name.lower().replace('/', '_')}.html"
            profile.to_file(output_path)
            print(f"  ✓ Saved: {output_path.name}")

        print(f"\n✓ Generated 4 class-specific HTML reports with word clouds")

    except ImportError:
        print("⚠ ydata-profiling not installed, skipping HTML report generation")
    except Exception as e:
        print(f"⚠ Error generating profiling report: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AG_NEWS Dataset - Exploratory Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full EDA with caching
  python eda/ag_news_eda.py --use-cache

  # Refresh cache and run full EDA
  python eda/ag_news_eda.py --refresh-cache

  # Skip specific steps
  python eda/ag_news_eda.py --skip vocabulary_patterns,profiling

  # Run only specific steps
  python eda/ag_news_eda.py --only overview,preprocessing,pca

Available steps:
  - overview: Dataset overview
  - raw_text: Raw text analysis
  - preprocessing: Text preprocessing exploration
  - vocabulary_patterns: Vocabulary and language patterns (SLOW)
  - tfidf: TF-IDF characteristics
  - class_patterns: Class-specific patterns (log-odds, discriminative terms)
  - correlation: Feature correlation analysis
  - pca: PCA analysis
  - profiling: Generate HTML profiling report (SLOW)
"""
    )

    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Use cached data if available (default: False)'
    )

    parser.add_argument(
        '--refresh-cache',
        action='store_true',
        help='Refresh all caches (ignores existing cache)'
    )

    parser.add_argument(
        '--skip',
        type=str,
        default='',
        help='Comma-separated list of steps to skip (e.g., "vocabulary_patterns,profiling")'
    )

    parser.add_argument(
        '--only',
        type=str,
        default='',
        help='Comma-separated list of steps to run (e.g., "overview,pca"). If specified, only these steps will run.'
    )

    return parser.parse_args()


def should_run_step(step_name, args):
    """Check if a step should be run based on CLI arguments"""
    # If --only is specified, only run those steps
    if args.only:
        only_steps = [s.strip() for s in args.only.split(',')]
        return step_name in only_steps

    # Otherwise, run all steps except those in --skip
    if args.skip:
        skip_steps = [s.strip() for s in args.skip.split(',')]
        return step_name not in skip_steps

    return True


def main():
    """Main execution function"""
    args = parse_args()

    # Determine cache usage
    use_cache = args.use_cache and not args.refresh_cache

    print("="*60)
    print("AG_NEWS DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    if use_cache:
        print("Using cache if available")
    if args.refresh_cache:
        print("Refreshing all caches")
    if args.skip:
        print(f"Skipping steps: {args.skip}")
    if args.only:
        print(f"Running only: {args.only}")

    # Initialize loader
    print("\nInitializing data loader...")
    loader = AGNewsLoader(max_features=10000, max_df=0.5, min_df=5)

    # Initialize statistics dictionary
    stats = {}

    # 1. Dataset Overview (always run - needed for other steps)
    train_texts, train_labels, test_texts, test_labels, overview_stats = plot_dataset_overview(loader, use_cache)
    stats.update(overview_stats)

    # 2. Raw Text Analysis
    if should_run_step('raw_text', args):
        analyze_raw_text(train_texts, train_labels, stats)

    # 3. Text Preprocessing Exploration (always run if tfidf or later steps are needed)
    if should_run_step('preprocessing', args) or should_run_step('tfidf', args) or should_run_step('correlation', args) or should_run_step('pca', args):
        vectorizer = analyze_preprocessing(train_texts, train_labels, stats)
    else:
        vectorizer = None

    # 3d. Vocabulary and Language Patterns
    if should_run_step('vocabulary_patterns', args):
        analyze_vocabulary_patterns(train_texts, stats, use_cache)

    # 3e. TF-IDF Characteristics
    if should_run_step('tfidf', args) and vectorizer is not None:
        X_tfidf = analyze_tfidf(train_texts, train_labels, vectorizer, stats)
    elif should_run_step('correlation', args) or should_run_step('pca', args) or should_run_step('class_patterns', args):
        # Need X_tfidf for later steps
        if vectorizer is None:
            vectorizer = analyze_preprocessing(train_texts, train_labels, stats)
        X_tfidf = analyze_tfidf(train_texts, train_labels, vectorizer, stats)
    else:
        X_tfidf = None

    # 3f. Class-Specific Patterns
    if should_run_step('class_patterns', args) and vectorizer is not None:
        analyze_class_specific_patterns(train_texts, train_labels, vectorizer, stats)

    # 4. Feature Correlation Analysis
    if should_run_step('correlation', args) and X_tfidf is not None:
        analyze_feature_correlation(X_tfidf, stats)

    # 5. PCA Analysis
    if should_run_step('pca', args) and X_tfidf is not None:
        analyze_pca(X_tfidf, train_labels, stats)

    # 6. Generate profiling report
    if should_run_step('profiling', args):
        generate_profiling_report(train_texts, train_labels)

    # Save statistics summary
    print("\n" + "="*60)
    print("SAVING STATISTICS SUMMARY")
    print("="*60)
    save_statistics(stats)

    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    if use_cache:
        print(f"Cache saved to: {CACHE_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
