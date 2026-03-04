"""
diabetes_ml/training/early_stopping.py
----------------------------------------
Implementação do Early Stopping com tolerância mínima de melhoria.

Classes
-------
EarlyStoppingState
    Estado mutável de um único modelo durante a busca.
EarlyStopping
    Lógica stateless que atualiza EarlyStoppingState a cada step.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field


@dataclass
class EarlyStoppingState:
    """Estado do Early Stopping para um único modelo."""

    best_acc: float = 0.0
    best_param: int = 1
    patience: int = 0
    active: bool = True
    test_acc: list[float] = field(default_factory=list)
    params: list[int] = field(default_factory=list)


class EarlyStopping:
    """
    Controla a parada antecipada com tolerância mínima de melhoria.

    Parameters
    ----------
    patience_limit : int
        Número máximo de steps consecutivos sem melhoria antes de parar.
    min_delta : float
        Melhoria mínima considerada significativa.
        Se a diferença ``acc - best_acc <= min_delta``, o passo é
        tratado como estagnação e a paciência é incrementada.

    ⚠️  ATENÇÃO — relação crítica com o tamanho do conjunto de teste
    ──────────────────────────────────────────────────────────────────
    A menor melhoria possível na acurácia é:

        resolução_mínima = 1 / n_amostras_teste

    Se min_delta ≥ resolução_mínima, nenhuma melhoria real conseguirá
    zerar a paciência e todos os modelos pararão após exatamente
    patience_limit steps. Use validate_against_dataset() logo após
    instanciar para detectar esse erro antes do treinamento começar.
    """

    def __init__(self, patience_limit: int, min_delta: float = 0.001) -> None:
        self.patience_limit = patience_limit
        self.min_delta = min_delta

    # ------------------------------------------------------------------
    # Validação preventiva
    # ------------------------------------------------------------------

    @classmethod
    def validate_against_dataset(
        cls,
        min_delta: float,
        n_total: int,
        test_size: float,
    ) -> None:
        """
        Verifica se min_delta é compatível com o tamanho do conjunto de teste.

        A menor melhoria possível na acurácia é 1/n_teste.
        Se min_delta >= 1/n_teste, a paciência nunca zerará e todos os
        modelos pararão após exatamente patience_limit steps.

        Parameters
        ----------
        min_delta : float  valor configurado em PipelineConfig
        n_total   : int    total de amostras no dataset
        test_size : float  fração de teste (ex: 0.3)

        Raises
        ------
        ValueError  se min_delta impossibilitar qualquer detecção de melhoria.
        """
        n_test = int(n_total * test_size)
        if n_test == 0:
            return

        resolution = 1.0 / n_test

        if min_delta >= resolution:
            raise ValueError(
                f"\n\n[EarlyStopping] Configuração INVÁLIDA detectada!\n"
                f"{'─' * 60}\n"
                f"  min_delta configurado : {min_delta:.6f}\n"
                f"  resolução mínima      : 1/{n_test} = {resolution:.6f}\n"
                f"  (dataset={n_total} amostras × test_size={test_size})\n\n"
                f"  PROBLEMA: min_delta >= resolução_mínima\n"
                f"  A acurácia nunca melhorará o suficiente para zerar\n"
                f"  a paciência. Todos os modelos pararão em:\n"
                f"    initial_param + patience_limit (sem exploração real).\n\n"
                f"{'─' * 60}\n"
                f"  ✅ Corrija definindo  min_delta < {resolution:.6f}\n"
                f"  ✅ Valor sugerido   : min_delta = {resolution / 3:.6f}\n"
            )

        if min_delta > resolution * 0.5:
            warnings.warn(
                f"[EarlyStopping] min_delta={min_delta:.6f} está acima de 50% da "
                f"resolução mínima ({resolution:.6f}). A busca pode perder melhorias "
                f"reais de apenas 1–2 amostras. Considere reduzir min_delta.",
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Lógica de stepping
    # ------------------------------------------------------------------

    def step(self, state: EarlyStoppingState, acc: float, param: int) -> None:
        """Registra o resultado do step e atualiza o estado."""
        state.test_acc.append(acc)
        state.params.append(param)

        improvement = acc - state.best_acc
        if improvement > self.min_delta:
            state.best_acc = acc
            state.best_param = param
            state.patience = 0
        else:
            state.patience += 1
            if state.patience >= self.patience_limit:
                state.active = False

    def is_active(self, state: EarlyStoppingState) -> bool:
        """Retorna True se o modelo ainda deve continuar sendo treinado."""
        return state.active