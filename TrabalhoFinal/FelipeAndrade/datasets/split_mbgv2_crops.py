import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS_DIR = os.path.join(BASE_DIR, "data", "crops")
METADATA_PATH = os.path.join(CROPS_DIR, "metadata.csv")

TRAIN_PATH = os.path.join(CROPS_DIR, "train.csv")
VAL_PATH = os.path.join(CROPS_DIR, "val.csv")
TEST_PATH = os.path.join(CROPS_DIR, "test.csv")

def main():
    print("Lendo metadata:", METADATA_PATH)
    df = pd.read_csv(METADATA_PATH)

    print("Total de amostras:", len(df))
    print("Classes únicas:", sorted(df["label"].unique()))

    # Split train vs (val+test) -> 70 / 30
    df_train, df_temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=42,
    )

    # Split val vs test -> 15 / 15
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        stratify=df_temp["label"],
        random_state=42,
    )

    print("Train:", len(df_train))
    print("Val:", len(df_val))
    print("Test:", len(df_test))

    df_train.to_csv(TRAIN_PATH, index=False)
    df_val.to_csv(VAL_PATH, index=False)
    df_test.to_csv(TEST_PATH, index=False)

    print("Salvo:")
    print("  ", TRAIN_PATH)
    print("  ", VAL_PATH)
    print("  ", TEST_PATH)

if __name__ == "__main__":
    main()
