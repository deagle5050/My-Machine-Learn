"""
diabetes_ml/visualization/gpu_canvas.py
-----------------------------------------
Gerencia os canvases vispy OpenGL para renderização 3D na GPU.

Design de sincronização
-----------------------
Todas as views compartilham **o mesmo objeto** TurntableCamera.
Como Python passa objetos por referência, qualquer rotação/zoom em
qualquer view se propaga instantaneamente para todas as outras —
sem eventos, sem callbacks, sem lag.

    cam = TurntableCamera(...)
    view_a.camera = cam   # ← referência compartilhada
    view_b.camera = cam   # ← mesmo objeto
    # girar view_a == girar view_b automaticamente
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from vispy import scene
from vispy.scene import visuals


# ── Helpers de cor ────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Converte '#RRGGBB' para tupla RGBA com valores 0–1."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


# ── Estado de uma view individual ─────────────────────────────────────────────

@dataclass
class ScatterViewState:
    """
    Contêiner de estado para um único ViewBox 3D.

    Guarda as referências dos vispy visuals (marcadores) e os metadados
    necessários para o tooltip de informação de pontos.
    """

    view: Any                        # vispy ViewBox

    # Metadados
    dataset_type: str  = ""          # 'train' | 'test'
    model_name:   str  = ""
    error_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )

    # Vispy Markers visuals
    bg_markers:  Any = field(default=None)   # fronteira de decisão (background)
    pt_markers:  Any = field(default=None)   # modo padrão
    err_markers: Any = field(default=None)   # modo de erros

    # Referências aos dados de posicionamento e cores (para atualizações dinâmicas)
    bg_pos:   np.ndarray = field(default_factory=lambda: np.array([]))
    bg_preds: np.ndarray = field(default_factory=lambda: np.array([]))
    pt_pos:   np.ndarray = field(default_factory=lambda: np.array([]))
    pt_preds: np.ndarray = field(default_factory=lambda: np.array([]))
    err_pos:  np.ndarray = field(default_factory=lambda: np.array([]))
    err_preds: np.ndarray = field(default_factory=lambda: np.array([]))

    # Referências aos dados originais (para tooltip)
    feat_raw_ref: Any = field(default=None)  # DataFrame com features não escalonadas
    target_ref:   Any = field(default=None)  # np.ndarray de targets
    pred_ref:     Any = field(default=None)  # np.ndarray de predições

    # ── Controle de modo ──────────────────────────────────────────────────────

    def set_diff_mode(self, active: bool, fog_enabled: bool = True) -> None:
        """Alterna entre visualização padrão e modo de erros."""
        _toggle = {
            self.bg_markers:  (not active) and fog_enabled,
            self.pt_markers:  not active,
            self.err_markers: active,
        }
        for vis, should_show in _toggle.items():
            if vis is not None:
                vis.visible = should_show

    def set_fog_visible(self, visible: bool) -> None:
        """Controla apenas a visibilidade da névoa (background)."""
        if self.bg_markers is not None:
            self.bg_markers.visible = visible


# ── Canvas de uma linha de gráficos ──────────────────────────────────────────

class GPUScatterRow:
    """
    Um vispy SceneCanvas com 1 linha × n_cols de ViewBoxes 3D.

    O parâmetro ``camera`` é injetado externamente para que múltiplos
    GPUScatterRow compartilhem a mesma câmera e fiquem sempre sincronizados.

    Parameters
    ----------
    n_cols : int
        Número de ViewBoxes (= número de modelos).
    camera : scene.cameras.TurntableCamera
        Câmera compartilhada. Todas as views usam esta referência.
    size : tuple[int, int] = (1400, 340)
        Tamanho inicial do canvas em pixels (largura, altura).
    """

    _BG_COLOR  = '#0b0b18'
    _BORDER_COL = '#252545'

    def __init__(
        self,
        n_cols: int,
        camera: scene.cameras.TurntableCamera,
        size: tuple[int, int] = (1400, 340),
    ) -> None:
        self.n_cols  = n_cols
        self._camera = camera

        self.canvas = scene.SceneCanvas(
            keys='interactive',
            bgcolor=self._BG_COLOR,
            size=size,
            show=False,
        )
        self._grid = self.canvas.central_widget.add_grid(spacing=4, margin=4)

        self.views: list[ScatterViewState] = []
        self._init_views()

    # ── Público ───────────────────────────────────────────────────────────────

    def set_diff_mode(self, active: bool, fog_enabled: bool = True) -> None:
        for sv in self.views:
            sv.set_diff_mode(active, fog_enabled)
        self.canvas.update()

    def set_fog_visible(self, visible: bool) -> None:
        for sv in self.views:
            sv.set_fog_visible(visible)
        self.canvas.update()

    @property
    def native(self) -> Any:
        """Widget Qt pronto para ser inserido num QLayout."""
        return self.canvas.native

    # ── Privado ───────────────────────────────────────────────────────────────

    def _init_views(self) -> None:
        for col in range(self.n_cols):
            view = self._grid.add_view(row=0, col=col, border_color=self._BORDER_COL)

            # ── Camera compartilhada — sincronização perfeita ─────────────
            view.camera = self._camera

            # Eixos XYZ sutis como referência espacial
            axis = visuals.XYZAxis(parent=view.scene)
            axis.transform = scene.transforms.STTransform(scale=(0.3, 0.3, 0.3))

            self.views.append(ScatterViewState(view=view))
