from typing import Any

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class ADSRScheduler(_LRScheduler):
    attack_duration: int
    decay_duration: int
    sustain_duration: int
    start_lr: float
    max_lr: float
    sustain_ratio: float
    end_lr: float
    release_decay: float

    def __init__(
        self,
        optimizer: Optimizer,
        attack_duration: int = 5,
        decay_duration: int = 0,
        sustain_duration: int = 0,
        start_lr: float = 5e-6,
        max_lr: float = 1e-5,
        sustain_ratio: float = 1.0,
        end_lr: float = 1e-6,
        release_decay: float = 0.8,
        last_epoch: int = -1,
    ):
        self.attack_duration = attack_duration
        self.decay_duration = decay_duration
        self.sustain_duration = sustain_duration
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.sustain_ratio = sustain_ratio
        self.end_lr = end_lr
        self.release_decay = release_decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self._calc_lr()
        return [lr for _ in self.optimizer.param_groups]

    def _calc_lr(self) -> float:
        last_epoch = self.last_epoch
        if last_epoch < self.attack_duration:
            slope = (self.max_lr - self.start_lr) / self.attack_duration
            return slope * last_epoch + self.start_lr
        last_epoch -= self.attack_duration

        sustain_lr = self.max_lr * self.sustain_ratio
        if last_epoch < self.decay_duration:
            slope = (sustain_lr - self.max_lr) / self.decay_duration
            return slope * last_epoch + self.max_lr
        last_epoch -= self.decay_duration

        if last_epoch < self.sustain_duration:
            return sustain_lr
        last_epoch -= self.sustain_duration

        return (
            sustain_lr - self.end_lr
        ) * self.release_decay ** last_epoch + self.end_lr
