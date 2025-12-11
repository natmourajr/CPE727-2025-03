"""
Funções auxiliares comuns
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from src.config import LOGS_DIR


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Configura logger para o projeto

    Args:
        name: Nome do logger
        log_file: Caminho do arquivo de log (opcional)
        level: Nível de logging

    Returns:
        logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed=42):
    """
    Fixa seed para reprodutibilidade

    Args:
        seed: Valor da seed
    """
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Timer:
    """Context manager para medir tempo de execução"""

    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time

    def get_elapsed(self):
        """Retorna tempo decorrido em segundos"""
        return self.elapsed


def save_metrics(metrics, filename, metrics_dir=None):
    """
    Salva métricas em arquivo JSON

    Args:
        metrics: Dicionário com métricas
        filename: Nome do arquivo
        metrics_dir: Diretório para salvar (usa METRICS_DIR por padrão)
    """
    if metrics_dir is None:
        from src.config import METRICS_DIR

        metrics_dir = METRICS_DIR

    filepath = Path(metrics_dir) / filename

    # Adicionar timestamp
    metrics["timestamp"] = datetime.now().isoformat()

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Métricas salvas em: {filepath}")


def load_metrics(filename, metrics_dir=None):
    """
    Carrega métricas de arquivo JSON

    Args:
        filename: Nome do arquivo
        metrics_dir: Diretório onde buscar (usa METRICS_DIR por padrão)

    Returns:
        Dicionário com métricas
    """
    if metrics_dir is None:
        from src.config import METRICS_DIR

        metrics_dir = METRICS_DIR

    filepath = Path(metrics_dir) / filename

    with open(filepath, "r") as f:
        metrics = json.load(f)

    return metrics


if __name__ == "__main__":
    # Teste das funções auxiliares
    print("Testando funções auxiliares...\n")

    # Testar logger
    log_file = LOGS_DIR / "test.log"
    logger = setup_logger("test", log_file)
    logger.info("Teste de logging")
    print(f"✓ Logger criado: {log_file}\n")

    # Testar seed
    set_seed(42)
    print("✓ Seed fixada\n")

    # Testar timer
    with Timer("teste") as timer:
        time.sleep(0.1)
    print(f"✓ Timer: {timer.get_elapsed():.3f}s\n")

    # Testar save/load metrics
    test_metrics = {"accuracy": 0.85, "f1_score": 0.83, "model": "test"}
    save_metrics(test_metrics, "test_metrics.json")

    loaded_metrics = load_metrics("test_metrics.json")
    print(f"✓ Métricas carregadas: {loaded_metrics}\n")

    print("✓ Funções auxiliares funcionando corretamente!")
