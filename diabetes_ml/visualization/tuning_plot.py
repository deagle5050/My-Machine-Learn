"""
diabetes_ml/visualization/tuning_plot.py
-----------------------------------------
Renderiza o gráfico de acurácia vs parâmetro (Fine Tuning) na linha
inferior do layout.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from diabetes_ml.training.early_stopping import EarlyStoppingState


class TuningPlotBuilder:
    """Constrói o subplot de Fine Tuning a partir dos estados do Early Stopping."""

    # Adicione entradas aqui ao registrar novos modelos em wrappers.py
    COLORS: dict[str, str] = {"KNN": "blue", "RF": "green", "GB": "red"}
    LABELS: dict[str, str] = {
        "KNN": "KNN (Test)",
        "RF": "Random Forest (Test)",
        "GB": "Gradient Boosting (Test)",
    }

    def build(
        self,
        ax: plt.Axes,
        states: dict[str, EarlyStoppingState],
        min_delta: float,
    ) -> None:
        """Plota as curvas de acurácia e marca os melhores parâmetros."""
        for key, state in states.items():
            color = self.COLORS.get(key, "gray")
            label = self.LABELS.get(key, key)

            ax.plot(state.params, state.test_acc, label=label, color=color, alpha=0.7)
            ax.plot(
                state.best_param, state.best_acc,
                marker="*", color="gold", markersize=15,
                markeredgecolor="black", linestyle="None",
            )
            ax.text(
                state.best_param, state.best_acc + 0.01,
                f"Best {key}", color=color, ha="center",
            )

        max_p = max(
            (s.params[-1] for s in states.values() if s.params), default=1
        )
        step = max(1, max_p // 20)
        ax.set_xticks(range(1, max_p + step, step))
        ax.set_title(
            f"NVIDIA GPU Fine Tuning — Accuracy vs Parameter "
            f"(Early Stopping | min_delta={min_delta})",
            fontsize=12,
        )
        ax.set_xlabel("Parameter Value (K / N Estimators)")
        ax.set_ylabel("Test Accuracy")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.7)
