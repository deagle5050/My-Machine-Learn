"""
diabetes_ml/visualization/interaction.py
------------------------------------------
Janela Qt principal que integra:
  - Dois GPUScatterRow (vispy OpenGL) para os gráficos 3D
  - Um FigureCanvas matplotlib para o gráfico de Fine Tuning
  - Labels Qt para títulos e acurácias
  - Botões de Reset, Modo Erros, Rotação e Névoa

Sincronização de câmera
-----------------------
A sincronização é garantida em nível de objeto Python: os dois canvases
vispy recebem a *mesma instância* de TurntableCamera. Qualquer interação
em qualquer canvas propaga imediatamente para todos os outros porque
compartilham o mesmo estado de câmera.

Picking (clique em pontos)
--------------------------
O clique detecado pelo evento mouse_press do canvas vispy projeta os
pontos 3D para coordenadas de tela via a transformada da câmera e
encontra o vizinho mais próximo por distância euclidiana 2D.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyQt — compatível com Qt5 e Qt6 via vispy's app abstraction
try:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSizePolicy, QFrame,
        QSlider, QCheckBox, QGroupBox, QFormLayout, QDialog
    )
    from PyQt5.QtCore import Qt, QTimer
except ImportError:
    from PyQt6.QtWidgets import (                               # type: ignore[no-redef]
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSizePolicy, QFrame,
        QSlider, QCheckBox, QGroupBox, QFormLayout, QDialog
    )
    from PyQt6.QtCore import Qt, QTimer                         # type: ignore[no-redef]

from diabetes_ml.data.dataset import ProcessedDataset
from diabetes_ml.training.early_stopping import EarlyStoppingState
from diabetes_ml.visualization.gpu_canvas import GPUScatterRow, ScatterViewState
from diabetes_ml.visualization.tuning_plot import TuningPlotBuilder
from diabetes_ml.visualization.subplots import ModelSubplotBuilder


# ── Estilos ───────────────────────────────────────────────────────────────────

_DARK_BG   = '#0b0b18'
_MID_BG    = '#11112a'
_ACCENT    = '#2a2a5a'
_BORDER    = '#3a3a7a'
_TEXT_CLR  = '#c8c8e8'

_LBL_STYLE = f"""
    QLabel {{
        color: {_TEXT_CLR};
        font-size: 11px;
        font-weight: bold;
        background-color: {_MID_BG};
        padding: 5px 10px;
        border-bottom: 1px solid {_BORDER};
    }}
"""

_BTN_STYLE = f"""
    QPushButton {{
        background-color: {_ACCENT};
        color: {_TEXT_CLR};
        border: 1px solid {_BORDER};
        border-radius: 6px;
        padding: 7px 24px;
        font-size: 13px;
        min-width: 140px;
    }}
    QPushButton:hover   {{ background-color: #3a3a7a; border-color: #6666cc; }}
    QPushButton:pressed {{ background-color: #1a1a3a; }}
"""

_INFO_STYLE = f"""
    QLabel {{
        color: {_TEXT_CLR};
        font-size: 11px;
        background-color: {_MID_BG};
        padding: 4px 12px;
        border-top: 1px solid {_BORDER};
    }}
"""


# ── Diálogo de Configurações ──────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Janela Qt (pop-up) para configurar parâmetros de renderização."""
    
    def __init__(self, parent: DiabetesMLWindow) -> None:
        super().__init__(parent)
        self.win = parent
        self.setWindowTitle("Visualization Settings")
        self.setStyleSheet(f"background-color: {_DARK_BG}; color: {_TEXT_CLR};")
        self.resize(400, 300)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        
        group = QGroupBox("Rendering Parameters")
        group.setStyleSheet(f"border: 1px solid {_BORDER}; padding-top: 15px; font-weight: bold;")
        form = QFormLayout(group)
        
        # Sliders para BG_ALPHA, BG_SIZE, PT_SIZE, ERR_SIZE
        # Usa o primeiro builder como referência de valores iniciais
        ref_builder = self.win.builders[0]
        self.sld_bg_alpha = self._create_slider(0, 100, int(ref_builder.bg_alpha * 100))
        self.sld_bg_size  = self._create_slider(1, 50, int(ref_builder.bg_size))
        self.sld_pt_size  = self._create_slider(1, 50, int(ref_builder.pt_size))
        self.sld_err_size = self._create_slider(1, 50, int(ref_builder.err_size))
        
        form.addRow("Background Alpha (%)", self.sld_bg_alpha)
        form.addRow("Background Size (px)", self.sld_bg_size)
        form.addRow("Point Size (px)",      self.sld_pt_size)
        form.addRow("Error Size (px)",      self.sld_err_size)
        
        layout.addWidget(group)
        
        self.chk_tuning = QCheckBox("Show Fine Tuning Plot")
        self.chk_tuning.setChecked(self.win._tuning_container.isVisible())
        self.chk_tuning.setStyleSheet("font-size: 13px;")
        self.chk_tuning.toggled.connect(self.win._on_toggle_tuning)
        layout.addWidget(self.chk_tuning)
        
        layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(_BTN_STYLE)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

    def _create_slider(self, min_val: int, max_val: int, init_val: int) -> QSlider:
        sld = QSlider(Qt.Horizontal)
        sld.setRange(min_val, max_val)
        sld.setValue(init_val)
        sld.valueChanged.connect(self._on_changed)
        return sld

    def _on_changed(self) -> None:
        # Atualiza todos os builders registrados
        for builder in self.win.builders:
            builder.update_render_params(
                bg_alpha=self.sld_bg_alpha.value() / 100.0,
                bg_size=float(self.sld_bg_size.value()),
                pt_size=float(self.sld_pt_size.value()),
                err_size=float(self.sld_err_size.value())
            )
        self.win.train_row.canvas.update()
        self.win.test_row.canvas.update()
        self.win.clean_row.canvas.update()


# ── Janela principal ──────────────────────────────────────────────────────────

class DiabetesMLWindow(QMainWindow):
    """
    Janela Qt que orquestra toda a visualização do pipeline.

    Parameters
    ----------
    train_row, test_row, clean_row : GPUScatterRow
        Canvases vispy com gráficos 3D já populados.
    all_views : list[ScatterViewState]
        Todas as views para controle de modo diff.
    dataset : ProcessedDataset
        Necessário para o tooltip de informação de pontos.
    tuning_states : dict[str, EarlyStoppingState]
        Para o gráfico de Fine Tuning.
    builders : list[ModelSubplotBuilder]
        Referências para atualizar parâmetros de renderização.
    model_names : list[str]
        Nomes curtos dos modelos.
    acc_train, acc_test, acc_clean : list[float]
        Acurácias finais para cada cenário.
    """

    def __init__(
        self,
        train_row:     GPUScatterRow,
        test_row:      GPUScatterRow,
        clean_row:     GPUScatterRow,
        all_views:     list[ScatterViewState],
        dataset:       ProcessedDataset,
        tuning_states: dict[str, EarlyStoppingState],
        min_delta:     float,
        builders:      list[ModelSubplotBuilder],
        model_names:   list[str],
        acc_train:     list[float],
        acc_test:      list[float],
        acc_clean:     list[float],
    ) -> None:
        super().__init__()
        self.train_row    = train_row
        self.test_row     = test_row
        self.clean_row    = clean_row
        self.all_views    = all_views
        self.dataset      = dataset
        self.builders     = builders
        # Para compatibilidade interna, define self.builder como o primeiro
        self.builder      = builders[0]
        self.is_diff_mode = False
        self.is_fog_on    = True
        self._info_text   = "Click on a point to see information"

        self.setWindowTitle('Diabetes ML — GPU Visualization (3x3 Grid)')
        self.setStyleSheet(f'background-color: {_DARK_BG};')
        self.resize(1440, 1080)

        # Controle de rotação
        self._rotation_timer = QTimer()
        self._rotation_timer.timeout.connect(self._on_rotate_step)
        self.is_rotating = False

        self._build_ui(tuning_states, min_delta, model_names, acc_train, acc_test, acc_clean)
        self._connect_picking()

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build_ui(
        self,
        tuning_states: dict[str, EarlyStoppingState],
        min_delta:     float,
        model_names:   list[str],
        acc_train:     list[float],
        acc_test:      list[float],
        acc_clean:     list[float],
    ) -> None:
        root = QWidget()
        root.setStyleSheet(f'background-color: {_DARK_BG};')
        self.setCentralWidget(root)

        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.setSpacing(2)

        # ── Linha 1: Treino Original ──────────────────────────────────────
        vlay.addLayout(self._title_row(model_names, acc_train, prefix='Train (Orig)'))
        vlay.addWidget(self.train_row.native, stretch=3)

        vlay.addWidget(self._separator())

        # ── Linha 2: Teste Original ───────────────────────────────────────
        vlay.addLayout(self._title_row(model_names, acc_test, prefix='Test (Orig)'))
        vlay.addWidget(self.test_row.native, stretch=3)

        vlay.addWidget(self._separator())

        # ── Linha 3: Dados Limpos (Outliers Removed) ──────────────────────
        vlay.addLayout(self._title_row(model_names, acc_clean, prefix='Test (Cleaned)'))
        vlay.addWidget(self.clean_row.native, stretch=3)

        # ── Fine Tuning ───────────────────────────────────────────────────
        vlay.addWidget(self._separator())
        self._tuning_container = self._tuning_widget(tuning_states, min_delta)
        vlay.addWidget(self._tuning_container, stretch=2)

        # ── Barra de informação ──────────────────────────────────────────
        self._info_label = QLabel(self._info_text)
        self._info_label.setStyleSheet(_INFO_STYLE)
        self._info_label.setAlignment(Qt.AlignCenter)
        vlay.addWidget(self._info_label)

        # ── Botões ────────────────────────────────────────────────────────
        vlay.addLayout(self._button_row())

    def _on_toggle_tuning(self, checked: bool) -> None:
        """Exibe ou oculta o gráfico de Fine Tuning."""
        self._tuning_container.setVisible(checked)

    def _title_row(
        self, model_names: list[str], accs: list[float], prefix: str
    ) -> QHBoxLayout:
        hlay = QHBoxLayout()
        hlay.setSpacing(2)
        hlay.setContentsMargins(0, 0, 0, 0)
        for name, acc in zip(model_names, accs):
            lbl = QLabel(f'{prefix}: {name}    Acc: {acc:.4f}')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(_LBL_STYLE)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            hlay.addWidget(lbl)
        return hlay

    def _tuning_widget(
        self,
        states:    dict[str, EarlyStoppingState],
        min_delta: float,
    ) -> FigureCanvas:
        fig = Figure(figsize=(14, 2.8), facecolor=_DARK_BG)
        ax  = fig.add_subplot(111, facecolor=_MID_BG)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18)

        # Estilo escuro para o gráfico matplotlib
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)
        ax.tick_params(colors=_TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(_TEXT_CLR)
        ax.yaxis.label.set_color(_TEXT_CLR)
        ax.title.set_color(_TEXT_CLR)
        ax.grid(color=_BORDER, linestyle=':', alpha=0.5)

        TuningPlotBuilder().build(ax, states, min_delta)

        canvas = FigureCanvas(fig)
        canvas.setStyleSheet(f'background-color: {_DARK_BG};')
        return canvas

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f'color: {_BORDER}; background-color: {_BORDER};')
        line.setFixedHeight(1)
        return line

    def _button_row(self) -> QHBoxLayout:
        hlay = QHBoxLayout()
        hlay.setContentsMargins(0, 6, 0, 6)

        self._reset_btn    = QPushButton('Reset View')
        self._rotate_btn   = QPushButton('Auto Rotate')
        self._fog_btn      = QPushButton('Disable Fog')
        self._settings_btn = QPushButton('Settings')
        self._diff_btn     = QPushButton('Show Errors')
        
        # Legenda com cores HTML
        self._legend = QLabel(
            '  <span style="color:#3d9bff">●</span> Class 0 Correct   '
            '  <span style="color:#ff4d4d">●</span> Class 1 Correct   '
            '  <span style="color:#00ffaa">●</span> Class 0 Error   '
            '  <span style="color:#ffee00">●</span> Class 1 Error'
        )
        self._legend.setStyleSheet(f'color: {_TEXT_CLR}; font-size: 11px;')

        for btn in (self._reset_btn, self._rotate_btn, self._fog_btn, self._settings_btn, self._diff_btn):
            btn.setStyleSheet(_BTN_STYLE)
            btn.setFixedHeight(38)

        self._reset_btn.clicked.connect(self._on_reset)
        self._rotate_btn.clicked.connect(self._on_toggle_rotation)
        self._fog_btn.clicked.connect(self._on_toggle_fog)
        self._settings_btn.clicked.connect(self._on_open_settings)
        self._diff_btn.clicked.connect(self._on_toggle_diff)

        hlay.addWidget(self._legend)
        hlay.addStretch()
        hlay.addWidget(self._reset_btn)
        hlay.addSpacing(12)
        hlay.addWidget(self._rotate_btn)
        hlay.addSpacing(12)
        hlay.addWidget(self._fog_btn)
        hlay.addSpacing(12)
        hlay.addWidget(self._settings_btn)
        hlay.addSpacing(12)
        hlay.addWidget(self._diff_btn)
        hlay.addSpacing(20)
        return hlay

    # ── Picking ───────────────────────────────────────────────────────────────

    def _connect_picking(self) -> None:
        """Conecta o evento de clique nos dois canvases vispy."""
        for row in (self.train_row, self.test_row, self.clean_row):
            row.canvas.events.mouse_press.connect(
                lambda e, r=row: self._on_canvas_click(e, r)
            )

    def _on_canvas_click(self, event: Any, row: GPUScatterRow) -> None:
        """
        Detecta o ponto mais próximo ao clique e exibe suas informações.
        """
        if event.button != 1:   # somente botão esquerdo
            return

        click_pos = np.array(event.pos[:2], dtype=np.float64)

        # Descobre qual ViewBox foi clicado
        target_sv = self._find_view_at(click_pos, row)
        if target_sv is None:
            return

        sv = target_sv
        features = (
            self.dataset.features_train if sv.dataset_type == 'train'
            else self.dataset.features_test
        )

        # Tenta projetar pontos 3D → tela
        try:
            tr = sv.view.scene.transform
            screen_pts = tr.map(features)   # (N, 4) homogêneas
            screen_2d  = screen_pts[:, :2]

            dists = np.linalg.norm(screen_2d - click_pos, axis=1)
            nearest = int(np.argmin(dists))

            if dists[nearest] > 30:   # threshold em pixels
                return

            self._show_point_info(sv, nearest)
        except Exception:
            pass   # picking não-essencial; falha silenciosa

    @staticmethod
    def _find_view_at(
        click_pos: np.ndarray, row: GPUScatterRow
    ) -> ScatterViewState | None:
        """Retorna o ScatterViewState cujo ViewBox contém a posição de clique."""
        for sv in row.views:
            rect = sv.view.rect
            x, y, w, h = rect.left, rect.bottom, rect.width, rect.height
            if x <= click_pos[0] <= x + w and y <= click_pos[1] <= y + h:
                return sv
        return None

    def _show_point_info(self, sv: ScatterViewState, idx: int) -> None:
        """Atualiza a barra de informação com os dados do ponto clicado."""
        real_idx = sv.error_indices[idx] if self.is_diff_mode else idx

        if real_idx >= len(sv.feat_raw_ref):
            return

        row     = sv.feat_raw_ref.iloc[real_idx]
        actual  = sv.target_ref[real_idx]
        pred    = sv.pred_ref[real_idx]
        correct = 'Yes' if actual == pred else 'No'
        split   = 'Train' if sv.dataset_type == 'train' else 'Test'

        self._info_label.setText(
            f'[{split} · {sv.model_name}]   '
            f'Insulin: {row["Insulin"]:.1f}   '
            f'Glucose: {row["Glucose"]:.1f}   '
            f'BMI: {row["BMI"]:.1f}   '
            f'Actual Class: {actual}   '
            f'Prediction: {pred}   '
            f'Correct: {correct}'
        )

    # ── Botões ────────────────────────────────────────────────────────────────

    def _on_reset(self) -> None:
        """Reseta a câmera compartilhada para a posição inicial."""
        cam = self.train_row.views[0].view.camera
        cam.elevation = 25.0
        cam.azimuth   = 45.0
        cam.fov       = 40.0
        cam.distance  = 4.5
        self.train_row.canvas.update()
        self.test_row.canvas.update()
        self.clean_row.canvas.update()

    def _on_toggle_rotation(self) -> None:
        """Inicia/para a rotação automática da câmera."""
        self.is_rotating = not self.is_rotating
        if self.is_rotating:
            self._rotation_timer.start(30)  # Aproximadamente 33 FPS
            self._rotate_btn.setText('Stop Rotation')
        else:
            self._rotation_timer.stop()
            self._rotate_btn.setText('Auto Rotate')

    def _on_rotate_step(self) -> None:
        """Incrementa o azimute da câmera para uma rotação lenta."""
        cam = self.train_row.views[0].view.camera
        cam.azimuth += 0.4
        self.train_row.canvas.update()
        self.test_row.canvas.update()
        self.clean_row.canvas.update()

    def _on_toggle_fog(self) -> None:
        """Alterna a visibilidade da névoa."""
        self.is_fog_on = not self.is_fog_on
        self.train_row.set_fog_visible(self.is_fog_on)
        self.test_row.set_fog_visible(self.is_fog_on)
        self.clean_row.set_fog_visible(self.is_fog_on)
        label = 'Enable Fog' if not self.is_fog_on else 'Disable Fog'
        self._fog_btn.setText(label)

    def _on_open_settings(self) -> None:
        """Abre o pop-up de configurações."""
        dialog = SettingsDialog(self)
        dialog.exec()

    def _on_toggle_diff(self) -> None:
        """Alterna entre modo padrão e modo de visualização de erros."""
        self.is_diff_mode = not self.is_diff_mode
        self.train_row.set_diff_mode(self.is_diff_mode, self.is_fog_on)
        self.test_row.set_diff_mode(self.is_diff_mode, self.is_fog_on)
        self.clean_row.set_diff_mode(self.is_diff_mode, self.is_fog_on)
        label = 'Standard View' if self.is_diff_mode else 'Show Errors'
        self._diff_btn.setText(label)
        self._info_label.setText(self._info_text)
