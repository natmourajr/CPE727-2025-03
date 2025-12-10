import torch
import pandas as pd
import json
import os

from utils.preprocessing import (
    load_raw_ratings, parse_netflix_ratings, filter_sparse,
    prepare_data, save_preprocessing_objects
)

from utils.dataloaders import (
    create_dataloaders,
    create_autorec_dataloader
)

from models.factory import create_model

# DMF
from training.train_dmf import train as train_dmf
from training.train_dmf import evaluate_dmf

# AutoRec
from training.train_ae import train_autorec as train_ae
from training.train_ae import evaluate_autorec


def main(model_type="dmf"):
    print("🚀 Iniciando pré-processamento...")

    # -------------------------------
    # 1) PRÉ-PROCESSAMENTO
    # -------------------------------
    df_raw = load_raw_ratings()
    df = parse_netflix_ratings(df_raw)
    df_filtered = filter_sparse(df)

    (
        train_users, train_movies, train_ratings,
        test_users, test_movies, test_ratings,
        num_users, num_movies
    ) = prepare_data(df_filtered)

    # Para DMF
    _, _, train_loader, test_loader = create_dataloaders(
        train_users, train_movies, train_ratings,
        test_users, test_movies, test_ratings,
        batch_size=256
    )

    # Salvar pré-processamento
    save_preprocessing_objects(
        df_raw=df_raw,
        df=df,
        df_filtered=df_filtered,
        train_users=train_users,
        train_movies=train_movies,
        train_ratings=train_ratings,
        test_users=test_users,
        test_movies=test_movies,
        test_ratings=test_ratings,
        num_users=num_users,
        num_movies=num_movies
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_type, num_users, num_movies, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------------
    # 2) TREINO ESPECÍFICO DO MODELO
    # -------------------------------
    if model_type == "dmf":
        print("\n🔥 Treinando DMF...")
        train_losses = train_dmf(model, train_loader, optimizer, device, epochs=3)
        rmse = evaluate_dmf(model, test_loader, device)

    elif model_type == "ae":
        print("\n🔥 Treinando AutoRec...")

        # Criar DF
        df_train = pd.DataFrame({
            "User": train_users,
            "Movie": train_movies,
            "Rating": train_ratings
        })

        df_test = pd.DataFrame({
            "User": test_users,
            "Movie": test_movies,
            "Rating": test_ratings
        })

        # Pivotagem
        train_matrix = df_train.pivot_table(
            index="User", columns="Movie", values="Rating", fill_value=0
        )

        test_matrix = df_test.pivot_table(
            index="User", columns="Movie", values="Rating", fill_value=0
        )

        # Criar dataloader
        train_loader = create_autorec_dataloader(
            train_matrix.values, batch_size=64
        )

        # Treinar
        train_losses = train_ae(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epochs=3
        )

        # Avaliação
        rmse = evaluate_autorec(model, test_matrix.values, device)


    else:
        raise ValueError(f"❌ Tipo de modelo desconhecido: {model_type}")

    # -------------------------------
    # 3) SALVAR CURVAS E MODELO
    # -------------------------------
    os.makedirs(f"training/curves/{model_type}", exist_ok=True)

    with open(f"training/curves/{model_type}/curves.json", "w") as f:
        train_losses = [float(x) for x in train_losses]
        rmse = float(rmse)

        json.dump(
            {"train_losses": train_losses, "rmse": rmse},
            f,
            indent=4
        )

    print(f"📈 Curvas salvas em training/curves/{model_type}/curves.json")

    torch.save(model.state_dict(), f"training/{model_type}_model.pth")
    print(f"✔ Modelo salvo em training/{model_type}_model.pth")


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "dmf"
    main(model_type)
