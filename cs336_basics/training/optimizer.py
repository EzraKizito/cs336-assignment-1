import math
from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch
from torch import nn


class ExampleSGD(torch.optim.Optimizer):
    def __init__(
        self, 
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]],
        lr = 1e-3
        ) -> None:
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: 
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data 
                p.data -= lr / math.sqrt(t+1) * grad 
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]], 
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0, # lambda
        eps: float = 1e-8,
        # LR Scheduling Parameters
        min_lr: float = 1e-4,
        total_steps: int = 10000,
        warmup_steps: int = 500
    ) -> None:

        defaults = {
            "lr": lr, 
            "betas": betas, 
            "weight_decay": weight_decay, 
            "eps": eps,
            "min_lr": min_lr,
            "base_lr": lr, 
            "total_steps": total_steps, 
            "warmup_steps": warmup_steps
        }

        super().__init__(params, defaults)
        self.current_step = 0
        self.current_lr = min_lr

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups: 
            self.current_step = int(self.current_step)
            group["lr"] = learning_rate_schedule(
                t=self.current_step, 
                a_max=group["base_lr"],
                a_min=group["min_lr"], 
                T_w=group["warmup_steps"],
                T_c=group["total_steps"]
            )
        # for now, we can assume all param groups have the same lr 
        self.current_lr = self.param_groups[0]["lr"]
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None: 
                    continue

                state = self.state[p]
                m = state.get("m")
                if m is None: 
                    m = torch.zeros_like(p)
                v = state.get("v", torch.zeros_like(p))
                if v is None: 
                    v = torch.zeros_like(p)
                t = state.get("t", 1)

                # Compute gradient
                grad = p.grad

                # Compute adjusted learning rate for iteration t
                lr_t = lr * math.sqrt(1 - beta_2 ** t) / (1 - beta_1**t)

                # Apply weight decay
                p.mul_(1.0 - weight_decay * lr)

                # Update moment estimates
                updated_m = beta_1 * m + (1 - beta_1) * grad 
                updated_v = beta_2 * v + (1 - beta_2) * (grad**2)

                # Apply moment adjusted updates
                step = updated_m / (torch.sqrt(updated_v) + eps)
                p.add_(step, alpha=-lr_t)

                # Update state
                state["t"] = t + 1
                state["m"] = updated_m
                state["v"] = updated_v
        self.current_step += 1
        return loss

    # Ensure state persistence across checkpointing
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["current_step"] = self.current_step
        return state_dict

    def load_state_dict(self, state_dict):
        self.current_step = state_dict.pop("current_step", 0)
        super().load_state_dict(state_dict)


def learning_rate_sweep_with_sgd():
    """Learning rate sweep experiments"""
    weights = torch.nn.Parameter(5*torch.randn((10,10)))

    lr_sweep = [1e1, 1e2, 1e3]
    tol = 1e-6
    for lr in lr_sweep:
        opt = ExampleSGD([weights], lr=lr)
        loss_list = []

        print(f"\nLR: {lr}\n")
        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(loss.cpu().item())
            loss_list.append(loss.cpu().item())
            loss.backward()
            opt.step()

        # CHECK DIVERGENCE
        diff = loss_list[-1] - loss_list[-2]
        print(f"LR {lr} with {diff} converges? {diff <= 0 and abs(diff) < tol}")  

def learning_rate_schedule(
    t: int,
    a_max: float,
    a_min: float,
    T_w: int, 
    T_c: int
) -> float: 
    if t < T_w: 
        a_t = (t / T_w) * a_max
    elif t > T_c:
        a_t = a_min 
    else: 
        a_t = a_min + 0.5*(1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))*(a_max - a_min)

    return a_t

@torch.no_grad()
def gradient_clipping(
        parameters: Iterable[nn.Parameter], 
        max_l2_norm: float
    ):
    # Compute norm
    grads = [p.grad for p in parameters if p.grad is not None]

    total_norm = math.sqrt(sum(torch.sum(g**2) for g in grads))
    eps = 1e-6
    clip_coef = max_l2_norm / (total_norm + eps)
    
    if total_norm > max_l2_norm:
        for p in parameters: 
            if p.grad is not None:
                p.grad.mul_(clip_coef)
        

def main():
    run_example_sgd_sweep = False
    if run_example_sgd_sweep:
        learning_rate_sweep_with_sgd()
    
if __name__ == "__main__":
    main()