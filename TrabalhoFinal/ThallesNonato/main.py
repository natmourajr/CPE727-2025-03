import torch

from utils.preprocessing import (
    load_raw_ratings, parse_netflix_ratings, filter_sparse,
    prepare_data, save_preprocessing_objects
)

from utils.dataloaders import create_dataloaders
from models.factory import create_model
from training.train_dmf import train
from training.train_dmf import evaluate_dmf


def main(model_type="dmf"):
    print("🚀 Iniciando pré-processamento...")

    df_raw = load_raw_ratings()
    df = parse_netflix_ratings(df_raw)
    df_filtered = filter_sparse(df)

    (
        train_users, train_movies, train_ratings,
        test_users, test_movies, test_ratings,
        num_users, num_movies
    ) = prepare_data(df_filtered)

    _, _, train_loader, test_loader = create_dataloaders(
        train_users, train_movies, train_ratings,
        test_users, test_movies, test_ratings,
        batch_size=256
    )

    # Salvar preprocessamento
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

    # ------------ AQUI ------------
    train_losses = train(model, train_loader, optimizer, device, epochs=3)
    rmse = evaluate_dmf(model, test_loader, device)
    # ------------------------------

    # Salvar curves
    import json, os
    os.makedirs(f"training/curves/{model_type}", exist_ok=True)

    with open(f"training/curves/{model_type}/curves.json", "w") as f:
        json.dump({"train_losses": train_losses, "rmse": rmse}, f, indent=4)

    print(f"📈 Curvas salvas em training/curves/{model_type}/curves.json")

    # Salvar modelo
    torch.save(model.state_dict(), f"training/{model_type}_model.pth")
    print(f"✔ Modelo salvo em training/{model_type}_model.pth")


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "dmf"
    main(model_type)
