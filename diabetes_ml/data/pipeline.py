"""
diabetes_ml/data/pipeline.py
-----------------------------
Responsável por carregar o CSV, dividir em treino/teste e escalonar.
Retorna um ProcessedDataset pronto para uso na GPU.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from diabetes_ml.config import PipelineConfig
from diabetes_ml.data.dataset import ProcessedDataset


class DataPipeline:
    """Carrega, divide e escala os dados de entrada."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.scaler = MinMaxScaler(feature_range=config.scaler_range)

    # ------------------------------------------------------------------
    # Público
    # ------------------------------------------------------------------

    def build(self) -> ProcessedDataset:
        """Executa o pipeline completo e retorna o dataset processado."""
        df = self._load()
        return self._split_and_scale(df)

    def build_cleaned(self, threshold: float = 2.0) -> ProcessedDataset:
        """
        Cria um dataset com outliers removidos (ex: fora de 2 desvios padrão).
        Remove aproximadamente os 5% mais extremos para permitir melhor aprendizado.
        """
        df = self._load()
        features = df[list(self.config.feature_columns)]
        
        # Filtro de outliers via Z-Score (Desvio Padrão)
        z_scores = np.abs((features - features.mean()) / features.std())
        # Mantém apenas linhas onde NENHUMA feature é outlier
        filtered_df = df[(z_scores < threshold).all(axis=1)].copy()
        
        print(f"[Data] Outlier removal: {len(df)} -> {len(filtered_df)} samples")
        return self._split_and_scale(filtered_df)

    # ------------------------------------------------------------------
    # Privado
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        path = self.config.file_path
        if not path.exists():
            sys.exit(f"[ERRO] Arquivo '{path}' não encontrado.")
        return pd.read_csv(path)

    def _split_and_scale(self, df: pd.DataFrame) -> ProcessedDataset:
        features = df[list(self.config.feature_columns)]
        target = df[self.config.target_column]

        f_train, f_test, t_train, t_test = train_test_split(
            features,
            target,
            shuffle=False,
            test_size=self.config.test_size,
        )

        f_train = f_train.reset_index(drop=True)
        f_test = f_test.reset_index(drop=True)
        t_train = t_train.reset_index(drop=True)
        t_test = t_test.reset_index(drop=True)

        f_train_scaled = self.scaler.fit_transform(f_train).astype(np.float32)
        f_test_scaled = self.scaler.transform(f_test).astype(np.float32)

        return ProcessedDataset(
            features_train=f_train_scaled,
            features_test=f_test_scaled,
            target_train=t_train.values.astype(np.int32),
            target_test=t_test.values.astype(np.int32),
            features_train_raw=f_train,
            features_test_raw=f_test,
        )
