from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


class RegressionMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | tuple[int, ...] = (128, 64),
        output_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # instantiate layers
        layers: list[nn.Module] = []
        in_features = input_size

        # generating layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

@dataclass
class MLPTrainingResult:
    model: RegressionMLP
    scaler: StandardScaler | None
    history: dict[str, list[float]]
    metrics: dict[str, float]
    predictions: pd.DataFrame


class MLPRegressorTrainer:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _df_to_tensor(df: pd.DataFrame | pd.Series) -> torch.Tensor:
        if isinstance(df, pd.Series):
            arr = df.to_numpy(dtype=np.float32).reshape(-1, 1)
        else:
            arr = df.to_numpy(dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32)

    def train(
        self,
        data: pd.DataFrame,
        target_column: str,
        hidden_sizes: list[int] | tuple[int, ...] = (128, 64),
        test_size: float = 0.3,
        random_seed: int = 42,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        scale_features: bool = True,
        dropout: float = 0.0,
        save_models: bool = False,
        save_path: str | Path = "./",
    ) -> MLPTrainingResult:
        
        # Ensuring save path is made
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Checking if target column is present
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not in dataframe")

        # Dropping invalid rows from
        clean_data = data.dropna(subset=[target_column]).copy()

        # Formatting relevant dfs
        feature_df = clean_data.drop(columns=[target_column, "SMILES"], errors="ignore")
        target_df = clean_data[[target_column]].copy()

        # Applying numerics and dropping n/a
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
        feature_df = feature_df.dropna(axis=1)

        nunique = feature_df.nunique(dropna=False)
        feature_df = feature_df.loc[:, nunique > 1]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df,
            target_df,
            test_size=test_size,
            random_state=random_seed,
        )

        # Scaling features
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns,
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns,
            )

        # Converting data to pytorch.tensors
        X_train_t = self._df_to_tensor(X_train).to(self.device)
        X_test_t = self._df_to_tensor(X_test).to(self.device)
        y_train_t = self._df_to_tensor(y_train).to(self.device)
        y_test_t = self._df_to_tensor(y_test).to(self.device)

        # Generate mode;
        model = RegressionMLP(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            output_size=1,
            dropout=dropout,
        ).to(self.device)

        # Loss function
        criterion = nn.MSELoss()

        # Optimiser
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        history = {"train_loss": [], "test_loss": []}
        batch_size = batch_size or len(X_train_t)

        for _ in range(epochs):
            model.train()

            # Randomly shuffles the training data for each epoch
            permutation = torch.randperm(X_train_t.size(0), device=self.device)

            epoch_loss = 0.0

            for i in range(0, X_train_t.size(0), batch_size):
                idx = permutation[i : i + batch_size]
                xb = X_train_t[idx]
                yb = y_train_t[idx]

                optimiser.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.detach().cpu()) * len(idx)

            epoch_loss /= X_train_t.size(0)

            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t)
                test_loss = float(criterion(test_preds, y_test_t).cpu())

            # Saving losses
            history["train_loss"].append(epoch_loss)
            history["test_loss"].append(test_loss)

        model.eval()
        with torch.no_grad():
            final_test_preds = model(X_test_t).cpu().numpy().reshape(-1)

        y_true = y_test.iloc[:, 0].to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y_true, final_test_preds)))
        r2 = float(r2_score(y_true, final_test_preds))

        predictions = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": final_test_preds,
            },
            index=y_test.index,
        )
        predictions.index.name = "ID"

        metrics = {
            "test_rmse": rmse,
            "test_r2": r2,
        }

        if save_models:
            torch.save(model.state_dict(), save_path / "mlp_model.pt")
            predictions.to_csv(save_path / "mlp_predictions.csv", index_label="ID")
            if scaler is not None:
                joblib.dump(scaler, save_path / "mlp_scaler.pkl")

        return MLPTrainingResult(
            model=model,
            scaler=scaler,
            history=history,
            metrics=metrics,
            predictions=predictions,
        )
