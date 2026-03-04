"""
diabetes_ml/config.py
---------------------
Configuração central e imutável de todo o pipeline.
Altere os valores aqui para ajustar o comportamento sem tocar em
nenhuma outra parte do código.

⚠️  RELAÇÃO CRÍTICA: test_size ↔ min_delta
─────────────────────────────────────────────────────────────────────
A menor melhoria possível na acurácia é determinada pelo tamanho do
conjunto de teste:

    resolução_mínima = 1 / n_amostras_teste
                     = 1 / (n_total × test_size)

Exemplos:
    2000 amostras, test_size=0.30  →  1/600  ≈ 0.0017  (0.17%)
    2000 amostras, test_size=0.10  →  1/200  ≈ 0.0050  (0.50%)

min_delta DEVE ser MENOR que essa resolução mínima.
Se min_delta ≥ resolução_mínima, a paciência nunca será zerada e
todos os modelos pararão após exatamente patience_limit steps.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Configuração imutável de todo o pipeline."""

    # Dados
    file_path: Path = Path("./diabetes.csv")
    feature_columns: tuple[str, ...] = ("Insulin", "Glucose", "BMI")
    target_column: str = "Outcome"
    test_size: float = 0.3
    random_state: int = 42
    scaler_range: tuple[float, float] = (-1.0, 1.0)

    # Early Stopping
    patience_limit: int = 150
    # min_delta deve ser MENOR que 1/n_amostras_teste.
    min_delta: float = 0.001
    initial_param: int = 20

    # Grade de fronteiras de decisão
    grid_resolution: int = 15
    grid_axis_range: tuple[float, float] = (-1.1, 1.1)