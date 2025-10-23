from typing import Dict, Any, Tuple, List
import math

import torch
from torch.optim import lr_scheduler

try:
    from torch.optim.lr_scheduler import PolynomialLR  # type: ignore
    _HAS_POLY = True
except Exception:
    _HAS_POLY = False


class Scheduler:
    """
    Scheduler: create PyTorch LR schedulers from a name + params.

    Supported schedulers:
    LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR,
    ExponentialLR, PolynomialLR (builtin if available, else Lambda fallback),
    CosineAnnealingLR, ChainedScheduler, SequentialLR, ReduceLROnPlateau,
    CyclicLR, OneCycleLR.

    The class returns the scheduler object already constructed and ready to be
    stepped in the training loop.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, num_epochs: int, steps_per_epoch: int):
        self.optimizer = optimizer
        self.num_epochs = max(1, int(num_epochs))
        self.steps_per_epoch = max(0, int(steps_per_epoch))
        self.total_steps = self.num_epochs * self.steps_per_epoch if self.steps_per_epoch > 0 else self.num_epochs

    def _default_warmup_then_main(self) -> Tuple[lr_scheduler._LRScheduler, lr_scheduler._LRScheduler, int]:
        """
        Build a small Linear warmup followed by Cosine annealing main schedule.
        Returns (warmup_scheduler, main_scheduler, warmup_iters)
        warmup_iters is expressed in iterations (total_steps).
        """
        warmup_iters = max(1, int(0.05 * self.total_steps)) if self.total_steps > 0 else min(5, self.num_epochs)
        warmup = lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_iters)
        # For main T_max expect epochs; convert warmup_iters to epochs if possible
        warmup_epochs = max(1, warmup_iters // max(1, self.steps_per_epoch)) if self.steps_per_epoch > 0 else warmup_iters
        T_max = max(1, self.num_epochs - warmup_epochs)
        main = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=1e-6)
        return warmup, main, warmup_iters

    def create(self, scheduler_name: str, scheduler_params: Dict[str, Any]) -> lr_scheduler._LRScheduler:
        """
        Create and return a scheduler instance based on name and params.
        """
        name = scheduler_name or 'CosineAnnealingLR'
        p = dict(scheduler_params or {})

        if name == 'LambdaLR':
            # lr_lambda must be callable; provide a sensible default if user didn't.
            lr_lambda = p.pop('lr_lambda', None)
            if not callable(lr_lambda):
                lr_lambda = lambda epoch: 1.0 / (1.0 + 0.1 * epoch)
            return lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        if name == 'MultiplicativeLR':
            factor = p.pop('factor', 0.95)
            lr_lambda = p.pop('lr_lambda', None)
            if not callable(lr_lambda):
                lr_lambda = (lambda epoch, f=factor: f)
            return lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lr_lambda)

        if name == 'StepLR':
            step_size = int(p.pop('step_size', 30))
            gamma = float(p.pop('gamma', 0.1))
            return lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        if name == 'MultiStepLR':
            milestones = p.pop('milestones', [30, 60, 90])
            gamma = float(p.pop('gamma', 0.1))
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        if name == 'ConstantLR':
            factor = float(p.pop('factor', 0.1))
            total_iters = int(p.pop('total_iters', max(1, int(0.05 * self.total_steps))))
            return lr_scheduler.ConstantLR(self.optimizer, factor=factor, total_iters=total_iters)

        if name == 'LinearLR':
            start_factor = float(p.pop('start_factor', 0.1))
            total_iters = int(p.pop('total_iters', max(1, int(0.05 * self.total_steps))))
            return lr_scheduler.LinearLR(self.optimizer, start_factor=start_factor, total_iters=total_iters)

        if name == 'ExponentialLR':
            gamma = float(p.pop('gamma', 0.95))
            return lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        if name == 'PolynomialLR':
            power = float(p.pop('power', 2.0))
            max_iter = int(p.pop('max_iter', self.total_steps if self.total_steps > 0 else self.num_epochs))
            if _HAS_POLY:
                return PolynomialLR(self.optimizer, total_iters=max_iter, power=power)
            else:
                def poly_lambda(epoch, max_iter=max_iter, power=power):
                    t = min(epoch, max_iter)
                    if t >= max_iter:
                        return 0.0
                    return (1.0 - float(t) / float(max_iter)) ** float(power)
                return lr_scheduler.LambdaLR(self.optimizer, lr_lambda=poly_lambda)

        if name == 'CosineAnnealingLR':
            T_max = int(p.pop('T_max', self.num_epochs))
            eta_min = float(p.pop('eta_min', 0.0))
            return lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)

        if name == 'ChainedScheduler':
            # For simplicity: chain a small Linear warmup and a Cosine main scheduler
            warmup, main, _ = self._default_warmup_then_main()
            return lr_scheduler.ChainedScheduler([warmup, main])

        if name == 'SequentialLR':
            warmup, main, warmup_iters = self._default_warmup_then_main()
            # SequentialLR expects milestones as integer list; use warmup_iters
            return lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup, main], milestones=[warmup_iters])

        if name == 'ReduceLROnPlateau':
            mode = p.pop('mode', 'min')
            factor = float(p.pop('factor', 0.1))
            patience = int(p.pop('patience', 5))
            threshold = float(p.pop('threshold', 1e-4))
            cooldown = int(p.pop('cooldown', 0))
            min_lr = float(p.pop('min_lr', 0.0))
            eps = float(p.pop('eps', 1e-8))
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience,
                                                  threshold=threshold, cooldown=cooldown, min_lr=min_lr, eps=eps)

        if name == 'CyclicLR':
            base_lr = float(p.pop('base_lr', self.optimizer.param_groups[0]['lr'] * 0.1))
            max_lr = float(p.pop('max_lr', self.optimizer.param_groups[0]['lr'] * 10.0))
            step_size_up = int(p.pop('step_size_up', max(1, math.floor(max(1, self.steps_per_epoch) / 2))))
            mode = p.pop('mode', 'triangular')
            cycle_momentum = bool(p.pop('cycle_momentum', False))
            return lr_scheduler.CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,
                                         mode=mode, cycle_momentum=cycle_momentum)

        if name == 'OneCycleLR':
            max_lr = float(p.pop('max_lr', self.optimizer.param_groups[0]['lr'] * 10.0))
            pct_start = float(p.pop('pct_start', 0.3))
            anneal_strategy = p.pop('anneal_strategy', 'cos')
            div_factor = float(p.pop('div_factor', 25.0))
            final_div_factor = float(p.pop('final_div_factor', 1e4))
            total_steps_param = p.pop('total_steps', None)
            total_steps_param = total_steps_param or self.total_steps
            return lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=total_steps_param,
                                           pct_start=pct_start, anneal_strategy=anneal_strategy,
                                           div_factor=div_factor, final_div_factor=final_div_factor)

        # Default fallback
        return lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
