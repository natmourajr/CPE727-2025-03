from models.dmf import DMF
from models.ae import AutoRec

def create_model(model_type, num_users, num_movies, device):
    model_type = model_type.lower()

    if model_type == "dmf":
        return DMF(num_users=num_users, num_movies=num_movies).to(device)

    if model_type == "ae":
        return AutoRec(num_items=num_movies).to(device)

    raise ValueError(f"Modelo desconhecido: {model_type}")
