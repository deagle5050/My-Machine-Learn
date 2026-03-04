"""
diabetes_ml/training/tuner.py
-------------------------------
Motor de busca de hiperparâmetros com Early Stopping.
Itera os modelos em paralelo (step a step) até todos pararem.
"""

from __future__ import annotations

import gc

import cupy as cp
import numpy as np

from diabetes_ml.config import PipelineConfig
from diabetes_ml.data.dataset import ProcessedDataset
from diabetes_ml.training.early_stopping import EarlyStopping, EarlyStoppingState
from diabetes_ml.training.wrappers import GPUModelWrapper


class HyperparameterTuner:
    """
    Executa a busca de hiperparâmetros com Early Stopping para
    múltiplos wrappers GPU em paralelo (step a step).

    Parameters
    ----------
    wrappers : list[GPUModelWrapper]
        Lista de modelos a buscar.
    dataset : ProcessedDataset
        Dados de treino e teste já na GPU.
    config : PipelineConfig
        Configuração global (patience, min_delta, param inicial).
    """

    def __init__(
        self,
        wrappers: list[GPUModelWrapper],
        dataset: ProcessedDataset,
        config: PipelineConfig,
    ) -> None:
        self.wrappers = wrappers
        self.dataset = dataset
        self.config = config
        self.early_stopping = EarlyStopping(
            patience_limit=config.patience_limit,
            min_delta=config.min_delta,
        )
        self.states: dict[str, EarlyStoppingState] = {
            w.name: EarlyStoppingState() for w in wrappers
        }

    @property
    def any_active(self) -> bool:
        """True enquanto ao menos um modelo ainda estiver em busca."""
        return any(s.active for s in self.states.values())

    def run(self) -> dict[str, EarlyStoppingState]:
        """
        Executa o loop de busca.

        Returns
        -------
        dict[str, EarlyStoppingState]
            Estados finais de cada modelo (inclui best_param e best_acc).
        """
        # Valida min_delta antes de gastar tempo de GPU
        n_total = (
            self.dataset.features_train.shape[0]
            + self.dataset.features_test.shape[0]
        )
        test_size = self.dataset.features_test.shape[0] / n_total
        EarlyStopping.validate_against_dataset(
            self.config.min_delta, n_total, test_size
        )

        print("Iniciando treinamento na GPU (NVIDIA CUDA) com Early Stopping...")
        param = self.config.initial_param

        while self.any_active:
            step_results: dict[str, float | None] = {}

            for wrapper in self.wrappers:
                state = self.states[wrapper.name]
                if not state.active:
                    step_results[wrapper.name] = None
                    continue

                acc = self._evaluate(wrapper, param)
                if acc is not None:
                    self.early_stopping.step(state, acc, param)
                step_results[wrapper.name] = acc

            self._log_step(param, step_results)

            if param % 10 == 0:
                gc.collect()

            param += 1

        print("\nBusca concluída por Early Stopping!")
        print("-" * 40)
        print(f"{'Modelo':<15} | {'Melhor Acc':<10} | {'Melhor Parâmetro':<15}")
        print("-" * 40)
        for name, state in self.states.items():
            print(f"{name:<15} | {state.best_acc:<10.4f} | {state.best_param:<15}")
        print("-" * 40)
        return self.states

    def _evaluate(self, wrapper: GPUModelWrapper, param: int) -> float | None:
        """Treina e avalia um modelo. Retorna None se o parâmetro for inválido."""
        n_samples = self.dataset.features_train.shape[0]
        if not wrapper.is_param_valid(param, n_samples):
            self.states[wrapper.name].active = False
            return None

        model = wrapper.build(param)
        model.fit(self.dataset.X_train_gpu, self.dataset.y_train_gpu)

        pred_gpu = model.predict(self.dataset.X_test_gpu)
        pred = cp.asnumpy(pred_gpu).astype(np.int32)
        return float(np.mean(pred == self.dataset.target_test))

    def _log_step(self, param: int, results: dict[str, float | None]) -> None:
        parts = [f"Step {param:3d}"]
        for wrapper in self.wrappers:
            state = self.states[wrapper.name]
            acc = results.get(wrapper.name)
            if acc is not None:
                parts.append(
                    f"{wrapper.name}: {acc:.4f} "
                    f"(Melhor: {state.best_acc:.4f} | "
                    f"Pac: {state.patience:2d}/{self.config.patience_limit})"
                )
            else:
                parts.append(
                    f"{wrapper.name} Parado (Melhor: {state.best_acc:.4f})"
                )
        print(" | ".join(parts))        