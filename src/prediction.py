# src/prediction.py

import os
from typing import Any, List, Optional

import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only


class PredictionWriter(Callback):
    """
    Callback to save test predictions to a CSV file.

    This callback collects predictions from each test batch and, at the end of the test epoch,
    saves them to a single CSV file, matching predictions with their corresponding IDs
    retrieved from the datamodule.
    """

    def __init__(self, output_dir: str, filename: str = "test_predictions.csv"):
        """
        Args:
            output_dir: The directory where the prediction file will be saved.
            filename: The name of the prediction file.
        """
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.predictions: List[torch.Tensor] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[torch.Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Gather predictions from each test batch."""
        if outputs is not None:
            self.predictions.append(outputs.detach().cpu())

    @rank_zero_only
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save predictions at the end of the test epoch."""
        if not self.predictions:
            print("No predictions to save.")
            return

        # Concatenate all predictions from test steps
        all_predictions = torch.cat(self.predictions).numpy().flatten()

        # Try to get test IDs from datamodule
        test_ids = getattr(trainer.datamodule, "test_ids", None)

        if test_ids is not None:
            if len(test_ids) != len(all_predictions):
                print(
                    f"Warning: Mismatch between number of test IDs ({len(test_ids)}) "
                    f"and predictions ({len(all_predictions)}). Saving without IDs."
                )
                df = pd.DataFrame({"predictions": all_predictions})
            else:
                df = pd.DataFrame({"id": test_ids, "tested_positive": all_predictions})
        else:
            print(
                "Warning: `test_ids` not found in datamodule. Saving predictions without IDs."
            )
            df = pd.DataFrame({"predictions": all_predictions})

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, self.filename)

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Test predictions saved to: {filepath}")

        # Clear the predictions list for the next run
        self.predictions.clear()
