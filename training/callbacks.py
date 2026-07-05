"""Callbacks — training loop me hooks (extensible).

Trainer key points par callbacks ko call karta hai. Built-in: EarlyStopping.
Naya behaviour add karna ho to Callback extend karo — trainer ko chhune ki
zaroorat nahi.
"""

from __future__ import annotations


class Callback:
    def on_train_begin(self, trainer): pass
    def on_step_end(self, trainer, step, metrics): pass
    def on_eval_end(self, trainer, step, metrics): pass
    def on_train_end(self, trainer): pass


class CallbackList:
    def __init__(self, callbacks=None):
        self.callbacks = list(callbacks or [])

    def add(self, cb: Callback):
        self.callbacks.append(cb)

    def on_train_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_step_end(self, trainer, step, metrics):
        for cb in self.callbacks:
            cb.on_step_end(trainer, step, metrics)

    def on_eval_end(self, trainer, step, metrics):
        for cb in self.callbacks:
            cb.on_eval_end(trainer, step, metrics)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    @property
    def should_stop(self) -> bool:
        return any(getattr(cb, "stop", False) for cb in self.callbacks)


class EarlyStopping(Callback):
    """N evals tak validation metric improve na ho to stop signal do."""

    def __init__(self, patience: int, monitor: str = "val_loss", min_delta: float = 0.0):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.stop = False

    def on_eval_end(self, trainer, step, metrics):
        if not self.patience:
            return
        value = metrics.get(self.monitor)
        if value is None:
            return
        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True
