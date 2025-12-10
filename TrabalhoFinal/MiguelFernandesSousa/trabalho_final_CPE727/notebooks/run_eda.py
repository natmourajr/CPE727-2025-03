"""
Exploratory Data Analysis for IARA Dataset.

Run this script to generate comprehensive analysis of the dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

from src.data.metadata import MetadataManager
from src.data.preprocessing import AudioPreprocessor, SpectrogramType

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def main():
    print("=" * 70)
    print("IARA Dataset - Exploratory Data Analysis")
    print("=" * 70)
    
    # Setup paths
    data_root = project_root / "data" / "downloaded_content"
    csv_path = project_root / "IARA" / "src" / "iara" / "dataset_info" / "iara.csv"
    xlsx_path = data_root / "iara.xlsx"
    
    # Initialize metadata manager
    print("\n1. Loading metadata...")
    manager = MetadataManager(
        csv_path=csv_path if csv_path.exists() else None,
        xlsx_path=xlsx_path if xlsx_path.exists() else None,
        data_root=data_root
    )
    
    df = manager.load_metadata()
    stats = manager.get_statistics()
    
    print(f"   ✓ Loaded {len(df)} recordings")
    
    # Class distribution
    print("\n2. Class Distribution:")
    class_dist = df['Class'].value_counts()
    for class_name, count in class_dist.items():
        pct = (count / len(df)) * 100
        print(f"   - {class_name:12s}: {count:4d} ({pct:5.2f}%)")
    
    # DC distribution
    print("\n3. Data Collection Distribution:")
    dc_dist = df['Dataset'].value_counts().sort_index()
    for dc, count in dc_dist.items():
        pct = (count / len(df)) * 100
        print(f"   - DC {dc}: {count:4d} ({pct:5.2f}%)")
    
    # Visualization
    print("\n4. Creating visualizations...")
    fig = plt.figure(figsize=(16, 12))
    
    # Class distribution bar plot
    ax1 = plt.subplot(2, 3, 1)
    class_dist.plot(kind='bar', ax=ax1, color=sns.color_palette("husl", len(class_dist)))
    ax1.set_title('Class Distribution', fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Recordings')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Class pie chart
    ax2 = plt.subplot(2, 3, 2)
    ax2.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("husl", len(class_dist)))
    ax2.set_title('Class Proportion', fontweight='bold')
    
    # DC distribution
    ax3 = plt.subplot(2, 3, 3)
    dc_dist.plot(kind='bar', ax=ax3, color=sns.color_palette("Set2", len(dc_dist)))
    ax3.set_title('Data Collection Distribution', fontweight='bold')
    ax3.set_xlabel('Data Collection')
    ax3.set_ylabel('Number of Recordings')
    ax3.grid(axis='y', alpha=0.3)
    
    # Vessel length distribution (for ships)
    ship_df = df[df['IsShip'] == True]
    if 'Length' in ship_df.columns and not ship_df['Length'].isna().all():
        ax4 = plt.subplot(2, 3, 4)
        ship_df['Length'].hist(bins=30, ax=ax4, edgecolor='black', alpha=0.7)
        ax4.axvline(50, color='r', linestyle='--', label='Small/Medium')
        ax4.axvline(100, color='orange', linestyle='--', label='Medium/Large')
        ax4.set_title('Vessel Length Distribution', fontweight='bold')
        ax4.set_xlabel('Length (m)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # Box plot by class
        ax5 = plt.subplot(2, 3, 5)
        ship_df.boxplot(column='Length', by='Class', ax=ax5)
        ax5.set_title('Length by Class', fontweight='bold')
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Length (m)')
        plt.sca(ax5)
        plt.xticks(rotation=45)
        
        print(f"\n   Vessel Length Statistics:")
        print(f"   - Mean: {ship_df['Length'].mean():.2f} m")
        print(f"   - Median: {ship_df['Length'].median():.2f} m")
        print(f"   - Min: {ship_df['Length'].min():.2f} m")
        print(f"   - Max: {ship_df['Length'].max():.2f} m")
    
    # Class by DC heatmap
    ax6 = plt.subplot(2, 3, 6)
    class_dc_crosstab = pd.crosstab(df['Dataset'], df['Class'])
    sns.heatmap(class_dc_crosstab, annot=True, fmt='d', cmap='YlGnBu', ax=ax6)
    ax6.set_title('Class Distribution by DC', fontweight='bold')
    ax6.set_xlabel('Class')
    ax6.set_ylabel('Data Collection')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "notebooks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eda_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved visualization to: {output_path}")
    
    # Analyze sample audio
    print("\n5. Analyzing sample audio from H folder...")
    h_folder = data_root / "H"
    if h_folder.exists():
        wav_files = list(h_folder.glob("*.wav"))
        if wav_files:
            sample_file = str(wav_files[0])
            sample_name = Path(sample_file).name
            
            # Load audio
            audio, sr = librosa.load(sample_file, sr=None)
            duration = len(audio) / sr
            
            print(f"   Sample: {sample_name}")
            print(f"   - Sample rate: {sr} Hz")
            print(f"   - Duration: {duration:.2f} seconds")
            print(f"   - Samples: {len(audio)}")
            
            # Extract spectrograms
            preprocessor = AudioPreprocessor(
                target_sr=16000,
                n_fft=1024,
                hop_length=1024,
                n_mels=128,
                averaging_windows=8,
            )
            
            audio_16k, _ = preprocessor.load_audio(sample_file)
            mel_spec = preprocessor.extract_mel_spectrogram(audio_16k)
            lofar_spec = preprocessor.extract_lofar_spectrogram(audio_16k)
            
            print(f"   ✓ MEL spectrogram shape: {mel_spec.shape}")
            print(f"   ✓ LOFAR spectrogram shape: {lofar_spec.shape}")
            
            # Visualize spectrograms
            fig, axes = plt.subplots(2, 1, figsize=(15, 8))
            
            librosa.display.specshow(
                mel_spec,
                sr=16000,
                hop_length=1024 * 8,
                x_axis='time',
                y_axis='mel',
                ax=axes[0],
                cmap='viridis'
            )
            axes[0].set_title(f'MEL Spectrogram - {sample_name}', fontweight='bold')
            axes[0].set_ylabel('Mel Frequency')
            plt.colorbar(axes[0].images[0], ax=axes[0], format='%+2.0f dB')
            
            librosa.display.specshow(
                lofar_spec,
                sr=16000,
                hop_length=1024 * 8,
                x_axis='time',
                y_axis='linear',
                ax=axes[1],
                cmap='viridis'
            )
            axes[1].set_title(f'LOFAR Spectrogram (TPSW) - {sample_name}', fontweight='bold')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Frequency (Hz)')
            plt.colorbar(axes[1].images[0], ax=axes[1])
            
            plt.tight_layout()
            spec_output = output_dir / "sample_spectrograms.png"
            plt.savefig(spec_output, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved spectrograms to: {spec_output}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total recordings: {len(df)}")
    print(f"  - UO: {stats['uo_recordings']}")
    print(f"  - Glider: {stats['glider_recordings']}")
    print(f"  - Ships: {stats['ship_recordings']}")
    print(f"  - Background: {stats['background_recordings']}")
    print("\nClass balance:")
    for class_name in manager.class_names:
        count = stats['class_distribution'].get(class_name, 0)
        pct = (count / len(df)) * 100
        balance = "⚠️  Imbalanced" if pct < 15 or pct > 40 else "✓ Balanced"
        print(f"  {class_name:12s}: {count:4d} ({pct:5.2f}%) {balance}")
    
    print("\n✓ EDA Complete!")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
