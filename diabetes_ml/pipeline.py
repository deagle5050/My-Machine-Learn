"""
diabetes_ml/pipeline.py
-------------------------
Orquestrador de alto nível do pipeline de ML.

Ordem de importação obrigatória
--------------------------------
vispy.app.use_app() deve ser chamado antes de qualquer importação Qt.
Por isso, a primeira coisa que este módulo faz é configurar o backend
vispy. O restante das importações segue depois.
"""

from __future__ import annotations

import sys

# ── Backend vispy — DEVE ser definido antes de qualquer import Qt ──────────
from vispy import app as _vispy_app
_vispy_app.use_app('pyqt5')   # troque por 'pyqt6' se o ambiente usar Qt6

# ── Agora é seguro importar Qt e matplotlib ────────────────────────────────
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt6.QtWidgets import QApplication         # type: ignore[no-redef]

import cupy as cp
import numpy as np
from vispy import scene

from diabetes_ml.config import PipelineConfig
from diabetes_ml.data import DataPipeline, ProcessedDataset
from diabetes_ml.training import (
    EarlyStoppingState,
    GPUModelWrapper,
    GradientBoostingWrapper,
    HyperparameterTuner,
    KNNWrapper,
    RandomForestWrapper,
)
from diabetes_ml.visualization import (
    DecisionBoundaryGrid,
    DiabetesMLWindow,
    GPUScatterRow,
    ModelSubplotBuilder,
    ScatterViewState,
)


class DiabetesMLPipeline:
    """
    Orquestrador de alto nível: conecta todas as camadas do projeto.

    Fluxo: dados → tuning (GPU) → melhores modelos → visualização (OpenGL GPU)

    Parameters
    ----------
    config : PipelineConfig, optional
        Configuração customizada. Usa valores padrão se omitida.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    # ── Público ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Executa o pipeline completo e abre a janela de visualização."""

        # 1. Dados (Original e Limpo)
        dp = DataPipeline(self.config)
        dataset_orig = dp.build()
        dataset_cln  = dp.build_cleaned()

        # 2. Busca de hiperparâmetros com Early Stopping (no original)
        wrappers = self._default_wrappers()
        states   = HyperparameterTuner(wrappers, dataset_orig, self.config).run()
        best_models_orig = self._build_best_models(wrappers, states)

        # 3. Grade de fronteira de decisão
        grid     = DecisionBoundaryGrid(self.config)
        grid_pos = np.column_stack((
            grid.flat_insulin, grid.flat_glucose, grid.flat_bmi
        )).astype(np.float32)

        # 4. Câmera compartilhada
        shared_camera = scene.cameras.TurntableCamera(
            fov=40.0, elevation=25.0, azimuth=45.0, distance=4.5
        )

        n_models  = len(best_models_orig)
        train_row = GPUScatterRow(n_models, shared_camera)
        test_row  = GPUScatterRow(n_models, shared_camera)
        clean_row = GPUScatterRow(n_models, shared_camera)

        # 5. Construção dos scatter plots
        builder     = ModelSubplotBuilder(train_row, test_row, dataset_orig)
        # Builder para a linha limpa (usa um dataset diferente)
        builder_cln = ModelSubplotBuilder(clean_row, clean_row, dataset_cln)
        
        all_views:  list[ScatterViewState] = []
        model_names: list[str] = []
        acc_trains:  list[float] = []
        acc_tests:   list[float] = []
        acc_cleans:  list[float] = []

        for col, (model_name, model_orig) in enumerate(best_models_orig.items()):
            # Recupera o wrapper e o parâmetro otimizado
            base_name = model_name.split(" (")[0]
            wrapper = next(w for w in wrappers if w.name == base_name)
            best_p = states[base_name].best_param
            
            # ── 1. Fluxo Original (Train & Test) ──────────────────────────
            model_orig.fit(dataset_orig.X_train_gpu, dataset_orig.y_train_gpu)
            pred_train_orig = cp.asnumpy(model_orig.predict(dataset_orig.X_train_gpu)).astype(np.int32)
            pred_test_orig  = cp.asnumpy(model_orig.predict(dataset_orig.X_test_gpu)).astype(np.int32)
            grid_preds_orig = cp.asnumpy(model_orig.predict(grid.X_grid_gpu)).astype(np.int32)

            acc_train = float(np.mean(pred_train_orig == dataset_orig.target_train))
            acc_test  = float(np.mean(pred_test_orig  == dataset_orig.target_test))

            sv_train, sv_test = builder.build(
                col, model_name, grid_pos, grid_preds_orig, pred_train_orig, pred_test_orig
            )
            
            # ── 2. Fluxo Dados Limpos (Instancia novo modelo) ──────────────
            model_cln = wrapper.build(best_p)
            model_cln.fit(dataset_cln.X_train_gpu, dataset_cln.y_train_gpu)
            
            pred_test_cln  = cp.asnumpy(model_cln.predict(dataset_cln.X_test_gpu)).astype(np.int32)
            grid_preds_cln = cp.asnumpy(model_cln.predict(grid.X_grid_gpu)).astype(np.int32)
            
            acc_cln = float(np.mean(pred_test_cln == dataset_cln.target_test))
            
            # Populamos apenas a visão de teste da clean_row (evitando duplicidade)
            sv_clean = clean_row.views[col]
            builder_cln._fill(sv_clean, model_name, 'test', grid_pos, grid_preds_cln, pred_test_cln)
            if sv_clean not in builder_cln._all_views:
                builder_cln._all_views.append(sv_clean)

            all_views.extend([sv_train, sv_test, sv_clean])
            model_names.append(model_name)
            acc_trains.append(acc_train)
            acc_tests.append(acc_test)
            acc_cleans.append(acc_cln)

        # 6. Qt Application + janela principal
        qt_app = QApplication.instance() or QApplication(sys.argv)

        window = DiabetesMLWindow(
            train_row     = train_row,
            test_row      = test_row,
            clean_row     = clean_row,
            all_views     = all_views,
            dataset       = dataset_orig,
            tuning_states = states,
            min_delta     = self.config.min_delta,
            builders      = [builder, builder_cln],
            model_names   = model_names,
            acc_train     = acc_trains,
            acc_test      = acc_tests,
            acc_clean     = acc_cleans,
        )
        window.show()
        qt_app.exec()

    # ── Privado ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_wrappers() -> list[GPUModelWrapper]:
        return [KNNWrapper(), RandomForestWrapper(), GradientBoostingWrapper()]

    @staticmethod
    def _build_best_models(
        wrappers: list[GPUModelWrapper],
        states:   dict[str, EarlyStoppingState],
    ) -> dict[str, object]:
        mapping = {w.name: w for w in wrappers}
        return {
            f"{name} (param={state.best_param})": mapping[name].build(state.best_param)
            for name, state in states.items()
        }

