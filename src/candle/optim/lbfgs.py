"""
LBFGS (Limited-memory BFGS) optimizer for candle.

Aligned with PyTorch's torch.optim.LBFGS.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._functional import (
    cat,
    dot,
    zeros_like,
)


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    This optimizer requires a closure that re-evaluates the model and
    returns the loss. Only a single parameter group is supported.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1).
        max_iter: Maximum number of iterations per optimization step (default: 20).
        max_eval: Maximum number of function evaluations per step.
            If None, set to max_iter * 5 / 4 (default: None).
        tolerance_grad: Termination tolerance on first order optimality (default: 1e-7).
        tolerance_change: Termination tolerance on function value/parameter
            changes (default: 1e-9).
        history_size: Update history size (default: 100).
        line_search_fn: Either 'strong_wolfe' or None (default: None).
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 1,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
    ):
        if max_eval is None:
            max_eval = int(max_iter * 5 / 4)

        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]["params"]
        self.state.setdefault("func_evals", 0)
        self.state.setdefault("n_iter", 0)

    def _gather_flat_grad(self):
        grads = []
        for p in self._params:
            if p.grad is None:
                grads.append(zeros_like(p).reshape(-1))
            else:
                grads.append(p.grad.detach().reshape(-1))
        return cat(grads, dim=0)

    def _gather_flat_params(self):
        return cat([p.detach().reshape(-1) for p in self._params], dim=0)

    def _set_flat_params(self, flat):
        offset = 0
        for p in self._params:
            numel = 1
            for s in p.shape:
                numel *= s
            p_flat = flat[offset:offset + numel].reshape(p.shape)
            p.data = p_flat
            offset += numel

    def _two_loop(self, flat_grad, old_dirs, old_stps, ro, H_diag):
        """L-BFGS two-loop recursion using tensor ops."""
        q = flat_grad.clone()
        num_old = len(old_dirs)
        alphas = [0.0] * num_old

        for i in range(num_old - 1, -1, -1):
            alphas[i] = float(dot(old_stps[i], q)) * ro[i]
            q = q - alphas[i] * old_dirs[i]

        r = q * H_diag

        for i in range(num_old):
            beta = float(dot(old_dirs[i], r)) * ro[i]
            r = r + (alphas[i] - beta) * old_stps[i]

        return r

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("LBFGS requires a closure")

        self._call_step_pre_hooks()

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]
        line_search_fn = group["line_search_fn"]

        state = self.state
        state.setdefault("old_dirs", [])
        state.setdefault("old_stps", [])
        state.setdefault("ro", [])
        state.setdefault("H_diag", 1.0)

        old_dirs = state["old_dirs"]
        old_stps = state["old_stps"]
        ro = state["ro"]

        current_loss = float(closure().detach().item())
        flat_grad = self._gather_flat_grad()
        n_eval = 1
        state["n_iter"] += 1

        for _ in range(max_iter):
            abs_grad_max = float(flat_grad.abs().amax().item())
            if abs_grad_max <= tolerance_grad:
                break

            d = self._two_loop(flat_grad, old_dirs, old_stps, ro, state["H_diag"])
            d = d * (-1.0)

            prev_flat_grad = flat_grad.clone()
            prev_loss = current_loss

            if line_search_fn == "strong_wolfe":
                x0 = self._gather_flat_params()
                step_size = self._line_search(closure, x0, d, current_loss, flat_grad)
                step = d * step_size
            else:
                step = d * lr

            self._set_flat_params(self._gather_flat_params() + step)
            current_loss = float(closure().detach().item())
            flat_grad = self._gather_flat_grad()
            n_eval += 1

            y = flat_grad - prev_flat_grad
            ys = float(dot(y, step).item())
            if ys > 1e-10:
                if len(old_dirs) >= history_size:
                    old_dirs.pop(0)
                    old_stps.pop(0)
                    ro.pop(0)
                old_dirs.append(y)
                old_stps.append(step)
                ro.append(1.0 / ys)
                state["H_diag"] = ys / float(dot(y, y).item())

            if abs(current_loss - prev_loss) < tolerance_change:
                break

            if n_eval >= max_eval:
                break

        state["func_evals"] += n_eval
        self._call_step_post_hooks()
        return current_loss

    def _line_search(self, closure, x0, d, f0, g0, c1=1e-4, max_ls=25):
        """Backtracking line search with Armijo condition."""
        dg0 = float(dot(g0, d).item())
        step_size = 1.0
        for _ in range(max_ls):
            x_new = x0 + step_size * d
            self._set_flat_params(x_new)
            f_new = float(closure().detach().item())
            if f_new <= f0 + c1 * step_size * dg0:
                return step_size
            step_size *= 0.5
        return step_size
