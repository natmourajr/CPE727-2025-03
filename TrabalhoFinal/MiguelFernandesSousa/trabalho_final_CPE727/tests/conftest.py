"""
Test configuration and fixtures.
"""

import pytest
from pathlib import Path

# Configure test paths
TEST_DATA_ROOT = Path("data/downloaded_content")
TEST_METADATA_CSV = Path("IARA/src/iara/dataset_info/iara.csv")
TEST_METADATA_XLSX = Path("data/downloaded_content/iara.xlsx")


@pytest.fixture
def data_root():
    """Provide data root path."""
    return TEST_DATA_ROOT


@pytest.fixture
def metadata_csv():
    """Provide metadata CSV path."""
    return TEST_METADATA_CSV


@pytest.fixture
def metadata_xlsx():
    """Provide metadata Excel path."""
    return TEST_METADATA_XLSX


@pytest.fixture
def sample_audio_file():
    """Provide path to a sample audio file from H folder."""
    h_folder = TEST_DATA_ROOT / "H"
    if not h_folder.exists():
        pytest.skip("H folder not found")
    
    wav_files = list(h_folder.glob("*.wav"))
    if not wav_files:
        pytest.skip("No WAV files in H folder")
    
    return str(wav_files[0])
