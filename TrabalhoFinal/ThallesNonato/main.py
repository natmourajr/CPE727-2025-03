import torch
import pandas as pd
import json
import os
from torch.utils.data import random_split, DataLoader
from utils.visualization import plot_learning_curve

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
from training.train_dmf import evaluate_ranking_dmf

# AutoRec
from training.train_ae import train_autorec as train_ae
from training.train_ae import evaluate_autorec

# NCF
from training.train_ncf import train_ncf
from training.train_ncf import evaluate_ncf
from training.train_ncf import evaluate_ranking


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

    # Para DMF e NCF
    _, _, train_loader_tmp, test_loader = create_dataloaders(
        train_users, train_movies, train_ratings,
        test_users, test_movies, test_ratings,
        batch_size=256
    )

    # Salvar pré-processamento
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_type, num_users, num_movies, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------------
    # Criar dataset/loader de validação (10% do treino)
    # -------------------------------
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_users, dtype=torch.long),
        torch.tensor(train_movies, dtype=torch.long),
        torch.tensor(train_ratings, dtype=torch.float32)
    )

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # -------------------------------
    # 2) TREINO ESPECÍFICO DO MODELO
    # -------------------------------
    if model_type == "dmf":
        print("\n🔥 Treinando DMF...")
        train_losses, val_losses = train_dmf(model, train_loader, optimizer, device, epochs=100, val_loader=val_loader)
        
        # Avaliação
        rmse = evaluate_dmf(model, test_loader, device)

        # Para Precision@K e NDCG@K, precisamos de uma matriz de usuário x item
        df_test = pd.DataFrame({
            "User": test_users,
            "Movie": test_movies,
            "Rating": test_ratings
        })

        MAX_TEST_ITEMS = 200

        # Reduzir para no máximo MAX_TEST_ITEMS por usuário
        df_test_reduced = df_test.groupby("User").apply(
            lambda x: x.sample(n=min(len(x), MAX_TEST_ITEMS), random_state=42)
        ).reset_index(drop=True)

        # Novos arrays
        test_users_reduced = df_test_reduced["User"].to_numpy()
        test_movies_reduced = df_test_reduced["Movie"].to_numpy()
        test_ratings_reduced = df_test_reduced["Rating"].to_numpy()

        precision, ndcg = evaluate_ranking_dmf(
            model,
            train_users, train_movies,
            test_users_reduced, test_movies_reduced,  # usar teste reduzido
            num_users, num_movies,
            device,
            K=10
        )


    elif model_type == "ncf":
        print("\n🔥 Treinando NCF...")
        train_losses, val_losses = train_ncf(
            model, 
            train_loader, 
            valid_loader=val_loader,
            epochs=100, 
            lr=0.001
        )

        # Avaliação RMSE
        rmse = evaluate_ncf(model, test_loader, device)

        # Criar DataFrame de teste
        df_test = pd.DataFrame({
            "User": test_users,
            "Movie": test_movies,
            "Rating": test_ratings
        })

        # Reduzir para no máximo MAX_TEST_ITEMS por usuário
        MAX_TEST_ITEMS = 200
        df_test_reduced = df_test.groupby("User").apply(
            lambda x: x.sample(n=min(len(x), MAX_TEST_ITEMS), random_state=42)
        ).reset_index(drop=True)

        test_users_reduced = df_test_reduced["User"].to_numpy()
        test_movies_reduced = df_test_reduced["Movie"].to_numpy()
        test_ratings_reduced = df_test_reduced["Rating"].to_numpy()

        # Avaliação de ranking
        precision, ndcg = evaluate_ranking(
            model,
            train_users, train_movies,
            test_users_reduced, test_movies_reduced,
            num_users, num_movies,
            device,
            K=10
        )

        print(f"Precision@10: {precision:.4f}, NDCG@10: {ndcg:.4f}")


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
        train_matrix = df_train.pivot_table(index="User", columns="Movie", values="Rating", fill_value=0)
        test_matrix  = df_test.pivot_table(index="User", columns="Movie", values="Rating", fill_value=0)

        # Criar dataloaders
        train_loader_ae = create_autorec_dataloader(train_matrix.to_numpy(), batch_size=64)
        val_loader_ae   = create_autorec_dataloader(test_matrix.to_numpy(), batch_size=64)

        # Treinar com validação
        train_losses, val_losses = train_ae(
            model=model,
            train_loader=train_loader_ae,
            optimizer=optimizer,
            device=device,
            epochs=2,
            val_loader=val_loader_ae
        )

        # Avaliação final
        rmse = evaluate_autorec(model, test_matrix.to_numpy(), device)

        # Métricas de ranking
        precision, ndcg = evaluate_ranking(model, test_matrix.to_numpy(), device, K=5)
        print(f"Precision@10: {precision:.4f}, NDCG@10: {ndcg:.4f}")

    elif model_type == "plot":
        plot_learning_curve()

    else:
        raise ValueError(f"❌ Tipo de modelo desconhecido: {model_type}")

    # -------------------------------
    # 3) SALVAR CURVAS E MODELO
    # -------------------------------
    os.makedirs(f"training/curves/{model_type}", exist_ok=True)

    with open(f"training/curves/{model_type}/curves.json", "w") as f:
        json.dump({
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "rmse": float(rmse)
        }, f, indent=4)

    print(f"📈 Curvas salvas em training/curves/{model_type}/curves.json")

    torch.save(model.state_dict(), f"training/{model_type}_model.pth")
    print(f"✔ Modelo salvo em training/{model_type}_model.pth")


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "dmf"
    main(model_type)
