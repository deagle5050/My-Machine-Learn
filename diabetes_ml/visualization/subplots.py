"""
diabetes_ml/visualization/subplots.py
---------------------------------------
Preenche os ScatterViewState com vispy Markers renderizados na GPU.

Cada ponto do scatter é enviado diretamente para a VRAM via OpenGL,
usando o mesmo buffer CuPy→NumPy já calculado pelo pipeline de treino.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from vispy.scene import visuals

from diabetes_ml.data.dataset import ProcessedDataset
from diabetes_ml.visualization.gpu_canvas import (
    GPUScatterRow,
    ScatterViewState,
    hex_to_rgba,
)


# ── Paleta de cores ───────────────────────────────────────────────────────────

_PALETTE: dict[str, str] = {
    'class_0': '#3d9bff',   # classe 0 correta  → azul
    'class_1': '#ff4d4d',   # classe 1 correta  → vermelho
    'error_0': '#00ffaa',   # classe 0 errada   → verde-menta
    'error_1': '#ffee00',   # classe 1 errada   → amarelo
}


# ── Helpers de cor ────────────────────────────────────────────────────────────

def _label_colors(labels: np.ndarray, kind: str, alpha: float = 1.0) -> np.ndarray:
    """Converte array de rótulos inteiros em (N, 4) RGBA float32."""
    out = np.empty((len(labels), 4), dtype=np.float32)
    for i, lbl in enumerate(labels):
        out[i] = hex_to_rgba(_PALETTE[f'{kind}_{int(lbl)}'], alpha)
    return out


def _mixed_colors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Cor de acerto para pontos corretos, cor de erro para incorretos."""
    out = np.empty((len(actual), 4), dtype=np.float32)
    for i, (a, p) in enumerate(zip(actual, predicted)):
        kind = 'class' if a == p else 'error'
        out[i] = hex_to_rgba(_PALETTE[f'{kind}_{int(a)}'])
    return out


# ── Construtor de subplots ────────────────────────────────────────────────────

class ModelSubplotBuilder:
    """
    Instancia os visuals vispy (Markers) para cada modelo nas linhas
    de treino e teste de dois GPUScatterRow.

    Os pontos são enviados para a VRAM via OpenGL na primeira chamada
    de set_data — todo o rendering subsequente ocorre na GPU.
    """

    def __init__(
        self,
        train_row: GPUScatterRow,
        test_row:  GPUScatterRow,
        dataset:   ProcessedDataset,
    ) -> None:
        self.train_row = train_row
        self.test_row  = test_row
        self.dataset   = dataset
        
        # Parâmetros de renderização dinâmicos
        self.bg_alpha = 0.4
        self.bg_size  = 12.0
        self.pt_size  = 7.0
        self.err_size = 8.5
        
        # Guardamos as referências para atualizações dinâmicas
        self._all_views: list[ScatterViewState] = []

    # ── Público ───────────────────────────────────────────────────────────────

    def update_render_params(
        self,
        bg_alpha: float | None = None,
        bg_size:  float | None = None,
        pt_size:  float | None = None,
        err_size: float | None = None,
    ) -> None:
        """Atualiza os parâmetros e reaplica aos marcadores existentes."""
        if bg_alpha is not None: self.bg_alpha = bg_alpha
        if bg_size  is not None: self.bg_size  = bg_size
        if pt_size  is not None: self.pt_size  = pt_size
        if err_size is not None: self.err_size = err_size

        for sv in self._all_views:
            if sv.bg_markers and sv.bg_pos.size > 0:
                bg_colors = _label_colors(sv.bg_preds, 'class', self.bg_alpha)
                sv.bg_markers.set_data(
                    pos=sv.bg_pos,
                    face_color=bg_colors,
                    size=self.bg_size,
                    edge_width=0
                )
            if sv.pt_markers and sv.pt_pos.size > 0:
                pt_colors = (
                    _label_colors(sv.pt_preds, 'class')
                    if sv.dataset_type == 'train'
                    else _mixed_colors(sv.pt_preds, sv.pred_ref)
                )
                sv.pt_markers.set_data(
                    pos=sv.pt_pos,
                    face_color=pt_colors,
                    size=self.pt_size,
                    edge_width=0
                )
            if sv.err_markers and sv.err_pos.size > 0:
                err_colors = _label_colors(sv.err_preds, 'error')
                sv.err_markers.set_data(
                    pos=sv.err_pos,
                    face_color=err_colors,
                    size=self.err_size,
                    edge_width=0
                )

    def build(
        self,
        col:        int,
        model_name: str,
        grid_pos:   np.ndarray,   # (N, 3) float32 — posições da malha
        grid_preds: np.ndarray,   # (N,)   int32   — predições na malha
        pred_train: np.ndarray,
        pred_test:  np.ndarray,
    ) -> tuple[ScatterViewState, ScatterViewState]:
        sv_train = self.train_row.views[col]
        sv_test  = self.test_row.views[col]

        self._fill(sv_train, model_name, 'train', grid_pos, grid_preds, pred_train)
        self._fill(sv_test,  model_name, 'test',  grid_pos, grid_preds, pred_test)
        
        if sv_train not in self._all_views: self._all_views.append(sv_train)
        if sv_test  not in self._all_views: self._all_views.append(sv_test)

        return sv_train, sv_test

    # ── Privado ───────────────────────────────────────────────────────────────

    def _fill(
        self,
        sv:          ScatterViewState,
        model_name:  str,
        split:       str,
        grid_pos:    np.ndarray,
        grid_preds:  np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        ds       = self.dataset
        is_train = (split == 'train')
        features = ds.features_train if is_train else ds.features_test
        targets  = ds.target_train   if is_train else ds.target_test
        raw_df   = ds.features_train_raw if is_train else ds.features_test_raw

        sv.dataset_type  = split
        sv.model_name    = model_name
        sv.feat_raw_ref  = raw_df
        sv.target_ref    = targets
        sv.pred_ref      = predictions

        error_mask       = targets != predictions
        sv.error_indices = np.where(error_mask)[0]

        # Salva dados para atualizações
        sv.bg_pos   = grid_pos.astype(np.float32)
        sv.bg_preds = grid_preds
        sv.pt_pos   = features.astype(np.float32)
        sv.pt_preds = targets
        
        err_feats  = features[error_mask]
        err_targets = targets[error_mask]
        sv.err_pos   = err_feats.astype(np.float32)
        sv.err_preds = err_targets

        # 1. Background: fronteira de decisão (névoa)
        bg_colors = _label_colors(grid_preds, 'class', self.bg_alpha)
        sv.bg_markers = self._markers(sv.view, grid_pos, bg_colors, self.bg_size)

        # 2. Pontos padrão
        pt_colors = (
            _label_colors(targets, 'class')
            if is_train
            else _mixed_colors(targets, predictions)
        )
        sv.pt_markers = self._markers(sv.view, features, pt_colors, self.pt_size)

        # 3. Pontos de erro (ocultos por padrão)
        err_colors = _label_colors(err_targets, 'error')
        sv.err_markers = self._markers(
            sv.view, err_feats, err_colors, self.err_size, visible=False
        )

    @staticmethod
    def _markers(
        view:       Any,
        pos:        np.ndarray,
        face_color: np.ndarray,
        size:       float,
        visible:    bool = True,
    ) -> visuals.Markers:
        """
        Cria um Markers visual e envia os dados para a GPU via OpenGL.
        Arrays vazios recebem um ponto fantasma com alpha=0.
        """
        if pos.shape[0] == 0:
            pos        = np.zeros((1, 3), dtype=np.float32)
            face_color = np.zeros((1, 4), dtype=np.float32)

        m = visuals.Markers(parent=view.scene)
        m.set_data(
            pos.astype(np.float32),
            face_color=face_color,
            edge_width=0,
            size=size,
        )
        m.visible = visible
        return m