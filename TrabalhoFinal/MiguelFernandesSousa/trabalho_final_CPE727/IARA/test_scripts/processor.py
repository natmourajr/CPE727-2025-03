"""
Processor Test Program

This script generate as images all processed data in a collection of IARA for test the processing
"""
import os
import shutil

import iara.records
import iara.processing.analysis as iara_proc
import iara.processing.manager as iara_manager
import iara.default as iara_default

def main(override: bool = True, only_sample: bool = False):
    """Main function print the processed data in a audio dataset."""

    proc_dir = "./data/processed_nan"

    if os.path.exists(proc_dir):
        shutil.rmtree(proc_dir)

    dp = iara_default.default_iara_lofar_audio_processor()
    # dp = iara_default.default_iara_mel_audio_processor()
    colletion = iara_default.default_collection(only_sample=only_sample)
    df = colletion.to_df()

    # for plot_type in iara_manager.PlotType:
    for plot_type in [iara_manager.PlotType.EXPORT_RAW, iara_manager.PlotType.EXPORT_PLOT]:
        dp.plot(df['ID'].head(10),
                plot_type=plot_type,
                frequency_in_x_axis=True,
                override=override)


if __name__ == "__main__":
    main()
