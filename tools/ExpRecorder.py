import os
import json
import time
import csv
from typing import Optional, Dict, Any
import torch

class ExpRecorder:
    def __init__(self, out_dir: str, main_metric: str = "val_f1", mode: str = "max"):
        """
        Args:
            out_dir: experiment root where models/, checkpoints/, and logs/ will be created.
            main_metric: primary metric used to select the best model, e.g., 'val_f1' or 'val_auc'.
            mode: 'max' means larger values are better; 'min' means smaller values are better.
        """
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.models_dir = os.path.join(self.out_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logs_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        self.history = []  # list of dicts per epoch
        self.best_metric_name = main_metric
        assert mode in ("max", "min")
        self.mode = mode
        self.best_metric_value = float("-inf") if mode == "max" else float("inf")
        self.best_ckpt_path = None
        self.start_time = time.time()

        self.csv_path = os.path.join(self.logs_dir, "history.csv")
        self.json_path = os.path.join(self.logs_dir, "history.json")
        self.meta_path = os.path.join(self.out_dir, "meta.json")

    def _is_better(self, metric_value: float) -> bool:
        if metric_value is None:
            return False
        if self.mode == "max":
            return metric_value > self.best_metric_value
        else:
            return metric_value < self.best_metric_value

    def save_epoch(self, epoch: int, metrics: Dict[str, Any],
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   meta: Optional[Dict[str, Any]] = None,
                   save_ckpt_always: bool = False):
        entry = {"epoch": int(epoch), "time": time.time() - self.start_time}
        entry.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
        self.history.append(entry)

        self._write_json()
        self._write_csv()

        if meta is not None:
            self._save_meta(meta)

        main_val = metrics.get(self.best_metric_name, None)
        if main_val is not None and self._is_better(main_val):
            self.best_metric_value = float(main_val)
            if model is not None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                ckpt_name = f"best_epoch{epoch}_{self.best_metric_name}={self.best_metric_value:.4f}_{timestamp}.pth"
                ckpt_path = os.path.join(self.models_dir, ckpt_name)
                self._save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, extra={"metrics": metrics})
                self.best_ckpt_path = ckpt_path
                print(f"[ExpRecorder] New best ({self.best_metric_name}={self.best_metric_value:.4f}) saved to {ckpt_path}")

        if save_ckpt_always and model is not None:
            ckpt_path = os.path。join(self.ckpt_dir, f"epoch_{epoch}.pth")
            self._save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, extra={"metrics": metrics})

    def _save_checkpoint(self, path: str, epoch: int, model: torch.nn.Module,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         scheduler: Optional[Any] = None,
                         extra: Optional[Dict[str, Any]] = None):
        obj = {
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
            "best_metric_name": self.best_metric_name,
            "best_metric_value": float(self.best_metric_value)
        }
        if optimizer is not None:
            obj["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            try:
                obj["scheduler_state"] = scheduler.state_dict()
            except Exception:
                obj["scheduler_state"] = None
        if extra is not None:
            obj["extra"] = extra
        torch.save(obj, path)

    def load_checkpoint(self, path: str, model: Optional[torch.nn.Module] = None,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        if model is not None and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "best_metric_value" in ckpt:
            self.best_metric_value = float(ckpt["best_metric_value"])
        return ckpt

    def _write_json(self):
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("[ExpRecorder] Failed to write json:", e)

    def _write_csv(self):
        try:
            import pandas as pd
            df = pd.DataFrame(self.history)
            df.to_csv(self.csv_path, index=False)
        except Exception:
            if not self.history:
                return
            keys = list(self.history[0].keys())
            try:
                with open(self.csv_path, "w", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for row in self.history:
                        writer.writerow(row)
            except Exception as e:
                print("[ExpRecorder] Failed to write csv:", e)

    def _save_meta(self, meta: Dict[str, Any]):
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("[ExpRecorder] Failed to save meta:", e)

    def get_history(self):
        return self.history

    def get_best_ckpt(self):
        return self.best_ckpt_path

    def summary(self):
        return {
            "n_epochs_recorded": len(self.history),
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "best_ckpt": self.best_ckpt_path
        }
