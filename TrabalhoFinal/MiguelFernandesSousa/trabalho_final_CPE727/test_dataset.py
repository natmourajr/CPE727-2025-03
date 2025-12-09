"""
Test script for IARA dataset and feature extractors.

Run with:
    python test_dataset.py --mode demo --data-dir /path/to/data --csv-path /path/to/iara.csv
    python test_dataset.py --mode unittest
"""

import argparse
import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Set PYTHONPATH for imports
os.environ['PYTHONPATH'] = str(src_path) + ':' + os.environ.get('PYTHONPATH', '')

# Import from src package
try:
    from src.features import LOFARExtractor, MELExtractor, resample_audio
    from src.dataset import IARADataset, create_iara_dataloaders
except ImportError:
    # Fallback for direct imports
    import features
    import dataset
    LOFARExtractor = features.LOFARExtractor
    MELExtractor = features.MELExtractor
    resample_audio = features.resample_audio
    IARADataset = dataset.IARADataset
    create_iara_dataloaders = dataset.create_iara_dataloaders


class TestFeatureExtractors(unittest.TestCase):
    """Unit tests for feature extractors."""

    def setUp(self):
        """Setup test audio signal."""
        # Create synthetic test signal
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Mix of frequencies (simulating underwater acoustic signal)
        signal = (
            np.sin(2 * np.pi * 100 * t) +  # 100 Hz
            0.5 * np.sin(2 * np.pi * 500 * t) +  # 500 Hz
            0.3 * np.sin(2 * np.pi * 1000 * t) +  # 1000 Hz
            0.1 * np.random.randn(len(t))  # Noise
        )

        self.test_audio = signal.astype(np.float32)
        self.sample_rate = sr

    def test_lofar_extractor_output_shape(self):
        """Test LOFAR extractor produces correct output shape."""
        extractor = LOFARExtractor(sr=self.sample_rate, n_fft=1024, hop_length=512)
        power, freqs, times = extractor.extract(self.test_audio)

        # Check dimensions
        self.assertEqual(power.ndim, 2, "Power should be 2D array")
        self.assertEqual(len(freqs), power.shape[0], "Frequency dimension mismatch")
        self.assertEqual(len(times), power.shape[1], "Time dimension mismatch")

    def test_lofar_extractor_torch(self):
        """Test LOFAR extractor torch output."""
        extractor = LOFARExtractor(sr=self.sample_rate, n_fft=1024, hop_length=512)
        features = extractor.extract_torch(self.test_audio)

        # Check tensor properties
        self.assertIsInstance(features, torch.Tensor, "Output should be torch.Tensor")
        self.assertEqual(features.ndim, 3, "Should have 3 dimensions [C, F, T]")
        self.assertEqual(features.shape[0], 1, "Should have 1 channel")

    def test_mel_extractor_output_shape(self):
        """Test MEL extractor produces correct output shape."""
        n_mels = 128
        extractor = MELExtractor(
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        power, freqs, times = extractor.extract(self.test_audio)

        # Check dimensions
        self.assertEqual(power.ndim, 2, "Power should be 2D array")
        self.assertEqual(power.shape[0], n_mels, f"Should have {n_mels} mel bands")
        self.assertEqual(len(freqs), n_mels, "Frequency bins mismatch")

    def test_mel_extractor_torch(self):
        """Test MEL extractor torch output."""
        extractor = MELExtractor(sr=self.sample_rate, n_fft=1024, hop_length=512)
        features = extractor.extract_torch(self.test_audio)

        # Check tensor properties
        self.assertIsInstance(features, torch.Tensor, "Output should be torch.Tensor")
        self.assertEqual(features.ndim, 3, "Should have 3 dimensions [C, F, T]")
        self.assertEqual(features.shape[0], 1, "Should have 1 channel")

    def test_audio_resampling(self):
        """Test audio resampling."""
        orig_sr = 44100
        target_sr = 16000

        # Create test signal at 44.1 kHz
        duration = 1.0
        t = np.linspace(0, duration, int(orig_sr * duration))
        signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        # Resample
        resampled = resample_audio(signal, orig_sr, target_sr)

        # Check length
        expected_length = int(len(signal) * target_sr / orig_sr)
        self.assertAlmostEqual(
            len(resampled),
            expected_length,
            delta=10,
            msg="Resampled length mismatch"
        )

    def test_feature_consistency(self):
        """Test that repeated extraction produces same features."""
        extractor = MELExtractor(sr=self.sample_rate)

        features1 = extractor.extract_torch(self.test_audio)
        features2 = extractor.extract_torch(self.test_audio)

        self.assertTrue(
            torch.allclose(features1, features2),
            "Repeated extraction should produce identical features"
        )


def demo_mode(data_dir: str, csv_path: str, num_samples: int = 3):
    """
    Demo mode: Load samples and visualize features.

    Args:
        data_dir: Path to IARA audio data
        csv_path: Path to IARA metadata CSV
        num_samples: Number of samples to visualize
    """
    print("=" * 80)
    print("IARA Dataset Demo Mode")
    print("=" * 80)

    try:
        # Create dataset
        print(f"\nLoading dataset from: {data_dir}")
        print(f"Metadata CSV: {csv_path}")

        dataset = IARADataset(
            data_dir=data_dir,
            csv_path=csv_path,
            feature_extractor='mel',
            target_type='size_class',
            data_collection=['A', 'B'],  # Use subset for faster demo
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )

        # Print dataset info
        info = dataset.get_info()
        print("\n" + "=" * 80)
        print("Dataset Information:")
        print("=" * 80)
        for key, value in info.items():
            print(f"{key}: {value}")

        # Test dataloaders
        print("\n" + "=" * 80)
        print("Creating DataLoaders...")
        print("=" * 80)

        train_loader, val_loader, test_loader, _ = create_iara_dataloaders(
            data_dir=data_dir,
            csv_path=csv_path,
            batch_size=4,
            feature_extractor='mel',
            target_type='size_class',
            data_collection=['A', 'B'],
            num_workers=0  # Use 0 for demo
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Load and visualize samples
        print("\n" + "=" * 80)
        print(f"Visualizing {num_samples} samples...")
        print("=" * 80)

        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(min(num_samples, len(dataset))):
            print(f"\nSample {i+1}:")

            # Get sample
            features, target, metadata = dataset[i]

            print(f"  File ID: {metadata['file_id']}")
            print(f"  Dataset: {metadata['dataset']}")
            print(f"  Vessel: {metadata['vessel_name']}")
            print(f"  Type: {metadata['vessel_type']}")
            print(f"  Length: {metadata['length']} m")
            print(f"  Target: {metadata['target_label']} (class {target})")
            print(f"  Feature shape: {features.shape}")

            # Extract both LOFAR and MEL for comparison
            # Re-load audio for LOFAR (this is just for demo)
            try:
                file_path = dataset._find_audio_file(metadata['file_id'])
                audio, sr = dataset._load_audio(file_path)
                audio = dataset._preprocess_audio(audio, sr)

                # LOFAR
                lofar_extractor = LOFARExtractor(sr=16000, n_fft=1024, hop_length=256)
                lofar_features = lofar_extractor.extract_torch(audio)

                # Plot LOFAR
                ax1 = axes[i, 0]
                im1 = ax1.imshow(
                    lofar_features[0].numpy(),
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                ax1.set_title(f'LOFAR - {metadata["target_label"]}')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Frequency')
                plt.colorbar(im1, ax=ax1)

                # Plot MEL (already extracted)
                ax2 = axes[i, 1]
                im2 = ax2.imshow(
                    features[0].numpy(),
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                ax2.set_title(f'MEL - {metadata["target_label"]}')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Mel Band')
                plt.colorbar(im2, ax=ax2)

            except Exception as e:
                print(f"  Warning: Could not visualize - {e}")

        plt.tight_layout()
        plt.savefig('iara_demo_features.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: iara_demo_features.png")

        # Test batch loading
        print("\n" + "=" * 80)
        print("Testing batch loading...")
        print("=" * 80)

        batch_features, batch_targets, batch_metadata = next(iter(train_loader))
        print(f"Batch features shape: {batch_features.shape}")
        print(f"Batch targets shape: {batch_targets.shape}")
        print(f"Batch targets: {batch_targets.tolist()}")

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError in demo mode: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Test IARA dataset and features')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['unittest', 'demo'],
        default='unittest',
        help='Test mode: unittest or demo'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Path to IARA audio data directory (required for demo mode)'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        help='Path to IARA metadata CSV (required for demo mode)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of samples to visualize in demo mode'
    )

    args = parser.parse_args()

    if args.mode == 'unittest':
        # Run unit tests
        print("Running unit tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatureExtractors)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        sys.exit(0 if result.wasSuccessful() else 1)

    elif args.mode == 'demo':
        # Run demo mode
        if not args.data_dir or not args.csv_path:
            print("Error: --data-dir and --csv-path are required for demo mode")
            sys.exit(1)

        demo_mode(args.data_dir, args.csv_path, args.num_samples)


if __name__ == '__main__':
    main()
