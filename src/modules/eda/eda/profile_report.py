"""Profile report generation using ydata-profiling

Wraps ydata-profiling to generate comprehensive EDA reports.
"""
import pandas as pd
from ydata_profiling import ProfileReport
from typing import Optional


def generate_profile_report(
    df: pd.DataFrame,
    title: str = "Dataset EDA Report",
    minimal: bool = False,
    explorative: bool = False,
) -> ProfileReport:
    """Generate a ydata-profiling report from a DataFrame

    Args:
        df: pandas DataFrame to analyze
        title: Report title
        minimal: If True, generates minimal report (faster)
        explorative: If True, generates detailed explorative report (slower)

    Returns:
        ProfileReport instance
    """
    if minimal:
        # Fast, minimal report
        config = {
            "title": title,
            "minimal": True,
            "samples": {"head": 5, "tail": 5},
            "correlations": None,
            "interactions": None,
            "missing_diagrams": None,
        }
    elif explorative:
        # Detailed explorative report
        config = {
            "title": title,
            "explorative": True,
        }
    else:
        # Default balanced report
        config = {
            "title": title,
            "correlations": {
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
            },
            "interactions": {"continuous": True},
            "samples": {"head": 10, "tail": 10},
        }

    report = ProfileReport(df, **config)
    return report


def generate_comparison_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    title: str = "Train/Test Comparison Report",
) -> tuple[ProfileReport, ProfileReport]:
    """Generate separate reports for train and test sets

    Args:
        train_df: Training set DataFrame
        test_df: Test set DataFrame
        title: Base title for reports

    Returns:
        Tuple of (train_report, test_report)
    """
    train_report = generate_profile_report(
        train_df,
        title=f"{title} - Training Set"
    )

    test_report = generate_profile_report(
        test_df,
        title=f"{title} - Test Set"
    )

    return train_report, test_report
