from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


class ValidatorPipeline:
    """
    Pipeline responsável por validar um experimento realizando:
      - Descoberta dos folds
      - Extração da melhor época (early stop)
      - Agregação das métricas entre folds (mean/std)
      - Agregação das curvas completas
      - Geração de gráficos mean ± std
      - Salvamento do relatório final
    """

    def __init__(self, experiment_path: Path, monitor: str = "map"):
        self.experiment_path = experiment_path
        self.monitor = monitor
        self.folds_dir = experiment_path / "folds"

        self.raw_curves = {}   # curvas completas de cada fold
        self.results = {}      # relatório final consolidado

    # ============================================================
    # API Principal
    # ============================================================
    def run(self):
        if not self.folds_dir.exists():
            raise FileNotFoundError(f"Pasta 'folds/' não encontrada: {self.folds_dir}")

        print(f"[INFO] Iniciando validação em: {self.experiment_path}")

        # 1. Carregar métricas de cada fold
        fold_best_metrics = self._iterate_folds()

        # 2. Agregar métricas finais
        aggregated_final = self._aggregate_metrics(fold_best_metrics)

        # 3. Agregar curvas completas
        aggregated_curves = self._aggregate_curves(self.raw_curves)

        # 4. Gerar gráficos
        self._save_plots(aggregated_curves)

        # 5. Salvar relatório final
        self.results = {
            "folds_best": fold_best_metrics,
            "aggregated_final": aggregated_final,
            "aggregated_curves": {
                metric: {
                    "mean": data["mean"].tolist(),
                    "std":  data["std"].tolist()
                }
                for metric, data in aggregated_curves.items()
            },
        }

        return self.results

    # ============================================================
    # Leitura dos arquivos
    # ============================================================
    def _load_metrics(self, metrics_file: Path) -> Dict[str, List[float]]:
        with open(metrics_file, "r") as f:
            return json.load(f)

    # ============================================================
    # Extração da melhor época por fold
    # ============================================================
    def _extract_best_epoch_metrics(self, metrics_dict: Dict[str, List[float]]) -> Dict[str, float]:

        if self.monitor not in metrics_dict:
            raise KeyError(f"Métrica monitorada '{self.monitor}' não encontrada no JSON.")

        monitor_curve = metrics_dict[self.monitor]
        best_epoch = int(np.argmax(monitor_curve))

        best_metrics = {
            metric_name: float(values[best_epoch])
            for metric_name, values in metrics_dict.items()
        }
        best_metrics["best_epoch"] = best_epoch
        return best_metrics

    # ============================================================
    # Iteração sobre os folds
    # ============================================================
    def _iterate_folds(self) -> Dict[str, Dict[str, float]]:
        fold_best = {}
        fold_raw = {}

        sorted_folds = sorted(self.folds_dir.iterdir())

        for fold_id, fold_path in enumerate(sorted_folds):

            metrics_path = fold_path / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Métricas não encontradas em: {metrics_path}")

            metrics_dict = self._load_metrics(metrics_path)

            # Salvar curvas completas
            fold_raw[f"fold_{fold_id}"] = metrics_dict

            # Melhor época
            best_metrics = self._extract_best_epoch_metrics(metrics_dict)
            fold_best[f"fold_{fold_id}"] = best_metrics

            print(f"[OK] Fold {fold_id}: best_epoch = {best_metrics['best_epoch']}")

        self.raw_curves = fold_raw
        return fold_best

    # ============================================================
    # Agregação das métricas finais (mean/std)
    # ============================================================
    def _aggregate_metrics(self, fold_results: Dict[str, Dict[str, float]]):
        print("\n[INFO] Calculando métricas agregadas (mean/std)…")

        metric_names = list(next(iter(fold_results.values())).keys())
        aggregated = {}

        for metric in metric_names:
            if metric == "best_epoch":
                continue

            values = np.array([fold_results[f][metric] for f in fold_results])

            aggregated[metric] = {
                "mean": float(values.mean()),
                "std":  float(values.std(ddof=1))
            }

            print(f" - {metric}: mean={aggregated[metric]['mean']:.4f}, std={aggregated[metric]['std']:.4f}")

        return aggregated

    # ============================================================
    # Agregação das curvas completas
    # ============================================================
    def _aggregate_curves(self, raw_curves: Dict[str, Dict[str, List[float]]]):
        print("\n[INFO] Agregando curvas completas entre folds…")

        metric_names = raw_curves[next(iter(raw_curves))].keys()
        output = {}

        # 1. Descobrir o menor comprimento entre os folds
        min_len = min(len(raw_curves[f][self.monitor]) for f in raw_curves)

        print(f"[INFO] Tamanho mínimo das curvas detectado: {min_len} épocas")

        for metric in metric_names:
            if metric == "best_epoch":
                continue

            # 2. Cortar todas as curvas para o tamanho mínimo
            trimmed = [
                np.array(raw_curves[f][metric][:min_len])
                for f in raw_curves
            ]

            stacks = np.stack(trimmed)

            output[metric] = {
                "mean": stacks.mean(axis=0),
                "std":  stacks.std(axis=0, ddof=1),
            }

        return output

    # ============================================================
    # Plot das curvas agregadas
    # ============================================================
    def _plot_curve(self, mean_curve, std_curve, title, save_path):
        epochs = np.arange(len(mean_curve))

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, mean_curve, label="Média", linewidth=2)
        plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.3, label="Desvio padrão")

        # Escala logarítmica para LR
        if title.lower() == "lr":
            plt.yscale("log")
            title += " log scale"

        plt.title(title)
        plt.xlabel("Época")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _save_plots(self, aggregated_curves):
        plots_dir = self.experiment_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        print("\n[INFO] Salvando gráficos em:", plots_dir)

        for metric, data in aggregated_curves.items():
            save_path = plots_dir / f"{metric}.png"

            self._plot_curve(
                mean_curve=data["mean"],
                std_curve=data["std"],
                title=metric,
                save_path=save_path
            )
            print(f"[PLOT] {metric} → {save_path}")

    def save_report(self):
        if not self.results:
            raise RuntimeError("Execute .run() antes de .save_report()")

        results_dir = self.experiment_path / "results"
        results_dir.mkdir(exist_ok=True)

        save_path = results_dir / "validation_report.json"
        with open(save_path, "w") as f:
            json.dump(self.results, f, indent=4)

        print(f"\n[INFO] Relatório salvo em: {save_path}")
