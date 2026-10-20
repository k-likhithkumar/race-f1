"""
Prediction Engine for Battle & Overtake Probability (M4 / Checkpoint 3C).

Provides:
- Point-in-time runtime prediction using trained regularized logistic regression pipelines.
- In-memory pipeline persistence (zero retraining on repeated API requests).
- Separation of battle probability and on-track overtake probability (never multiplied).
- Handling of missing causal features without fabricating values.
- Empirical reliability warnings supported by backtesting evidence.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.race.config import MODELS_DIR, ensure_directories
from src.race.analytics.features import (
    PairFeatures,
    extract_pair_features,
)
from src.race.analytics.model_training import (
    PREDICTION_FEATURES,
    create_logistic_pipeline,
)
from src.race.data.cache import load_cached_table

logger = logging.getLogger(__name__)


def train_and_save_default_models(model_dir: Path) -> Tuple[Pipeline, Pipeline]:
    """
    Trains standard battle and overtake pipelines on the validated Monza dataset
    (2022 and 2023 Monza, 855 valid samples) and serializes them to model_dir.
    """
    from src.race.analytics.features import build_overtake_dataset

    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training default prediction models and saving to {model_dir}...")

    laps_2022 = load_cached_table(2022, "monza", "laps")
    stints_2022 = load_cached_table(2022, "monza", "stints")
    laps_2023 = load_cached_table(2023, "monza", "laps")
    stints_2023 = load_cached_table(2023, "monza", "stints")

    races = [
        {"race_id": "2022_monza", "laps": laps_2022, "stints": stints_2022},
        {"race_id": "2023_monza", "laps": laps_2023, "stints": stints_2023},
    ]

    dataset_df = build_overtake_dataset(races, candidate_gap_threshold_s=3.0, horizon_laps=5)
    valid_df = dataset_df[dataset_df["label_valid"] == True].copy()

    if len(valid_df) == 0:
        raise ValueError("Cannot train models: valid dataset is empty.")

    X = valid_df[PREDICTION_FEATURES]
    y_battle = valid_df["battle_next_H_laps"].astype(int).values
    y_pass = valid_df["pass_next_H_laps"].astype(int).values

    battle_pipe = create_logistic_pipeline(c_regularization=1.0, random_state=42)
    battle_pipe.fit(X, y_battle)

    pass_pipe = create_logistic_pipeline(c_regularization=1.0, random_state=42)
    pass_pipe.fit(X, y_pass)

    joblib.dump(battle_pipe, model_dir / "battle_pipeline.joblib")
    joblib.dump(pass_pipe, model_dir / "overtake_pipeline.joblib")
    logger.info(f"Successfully serialized battle and overtake pipelines to {model_dir}")

    return battle_pipe, pass_pipe


class PredictionEngine:
    """
    Runtime prediction engine managing model artifacts, causal feature evaluation,
    and probabilistic inference.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or MODELS_DIR
        self._battle_pipeline: Optional[Pipeline] = None
        self._overtake_pipeline: Optional[Pipeline] = None
        self._models_loaded: bool = False

    def is_ready(self) -> bool:
        """Checks if both models are loaded in memory, attempting lazy loading if not."""
        if self._models_loaded:
            return True
        return self.load_models(allow_training=True)

    def load_models(self, allow_training: bool = True) -> bool:
        """
        Loads pipelines from disk into memory. If files do not exist and allow_training=True,
        trains default models once and persists them.
        """
        battle_path = self.model_dir / "battle_pipeline.joblib"
        overtake_path = self.model_dir / "overtake_pipeline.joblib"

        if battle_path.exists() and overtake_path.exists():
            try:
                self._battle_pipeline = joblib.load(battle_path)
                self._overtake_pipeline = joblib.load(overtake_path)
                self._models_loaded = True
                logger.info(f"Loaded existing prediction pipelines from {self.model_dir}")
                return True
            except Exception as exc:
                logger.error(f"Failed to load pipelines from {self.model_dir}: {exc}")
                self._models_loaded = False
                return False

        if allow_training:
            try:
                b_pipe, p_pipe = train_and_save_default_models(self.model_dir)
                self._battle_pipeline = b_pipe
                self._overtake_pipeline = p_pipe
                self._models_loaded = True
                return True
            except Exception as exc:
                logger.error(f"Failed to train and save models: {exc}")
                self._models_loaded = False
                return False

        self._models_loaded = False
        return False

    def unload_models(self) -> None:
        """Clears models from memory (used for testing MODEL_UNAVAILABLE behavior)."""
        self._battle_pipeline = None
        self._overtake_pipeline = None
        self._models_loaded = False

    def predict_pair(self, features: PairFeatures) -> Tuple[float, float]:
        """
        Computes separate probabilities for battle and overtake.

        Returns
        -------
        tuple of (float, float)
            (p_battle_next_5_laps, p_pass_next_5_laps)
        """
        if not self._models_loaded or self._battle_pipeline is None or self._overtake_pipeline is None:
            raise RuntimeError("Prediction models are not loaded.")

        row = features.to_dict()
        df = pd.DataFrame([row])[PREDICTION_FEATURES]

        p_battle = float(self._battle_pipeline.predict_proba(df)[0, 1])
        p_overtake = float(self._overtake_pipeline.predict_proba(df)[0, 1])

        return round(p_battle, 4), round(p_overtake, 4)


# Global singleton instance for application runtime
prediction_engine = PredictionEngine()
