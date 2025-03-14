# highly inspired by:
# - https://github.com/nsmith-/jaxfit/blob/roofit/src/jaxfit/minimize.py
# - https://github.com/patrick-kidger/optimistix/blob/main/optimistix/_solver/bfgs.py

import typing as tp

import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from equinox.internal import ω
import optimistix as optx

# internal optimistix imports
from optimistix._misc import (
    max_norm,
    cauchy_termination,
    tree_full_like,
    filter_cond,
    verbose_print,
    tree_dot,
)
from optimistix._minimise import AbstractMinimiser
from optimistix._custom_types import Aux, DescentState, Fn, SearchState, Y
from optimistix._search import FunctionInfo, AbstractDescent, AbstractSearch
from optimistix._solution import RESULTS
from optimistix._solver.bfgs import _identity_pytree
from optimistix._solver.backtracking import BacktrackingArmijo
from optimistix._solver.gauss_newton import NewtonDescent

from jaxtyping import Array, Bool, Int, PyTree, Scalar


# TODO: use as termination criterion
def _edm(hessian, hessian_inv, grad) -> jax.Array:
    if hessian is None:
        assert hessian_inv is not None
        return tree_dot(grad, hessian_inv.mv(grad)) / 2
    else:
        assert hessian_inv is None
        # we need to invert the hessian to get the edm, how?
        # code from zfit:
        #
        # invhessgrad = np.linalg.solve(
        #     hessian,
        #     grad,
        # )
        # return grad @ invhessgrad / 2
        raise NotImplementedError()


def _init_hessian_inv(fn, y, args):
    pytree = (1.0 / eqx.filter_hessian(fn)(y, args)[0] ** ω).ω
    return lx.PyTreeLinearOperator(
        pytree,
        output_structure=jax.eval_shape(lambda: y),
        tags=lx.positive_semidefinite_tag,
    )


def _init_hessian(fn, y, args):
    pytree = eqx.filter_hessian(fn)(y, args)[0]
    return lx.PyTreeLinearOperator(
        pytree,
        output_structure=jax.eval_shape(lambda: y),
        tags=lx.positive_semidefinite_tag,
    )


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jax.tree.map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jax.tree.map(leaf_fn, tree1)


# TODO: is this correct?
def _dfp_update(f_eval, grad, prev_grad, hessian, hessian_inv, y_diff) -> jax.Array:
    grad_diff = (grad**ω - prev_grad**ω).ω
    inner = tree_dot(grad_diff, y_diff)

    def dfp_update(hessian, hessian_inv):
        # https://seal.web.cern.ch/seal/documents/minuit/mntutorial.pdf, page 26
        # https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula
        if hessian is None:
            # Inverse Hessian
            assert hessian_inv is not None
            # DFP update to the operator directly
            inv_mvp = hessian_inv.mv(grad_diff)
            term1 = (_outer(y_diff, y_diff) ** ω / inner).ω
            term2 = (_outer(inv_mvp, inv_mvp) ** ω / tree_dot(grad_diff, inv_mvp)).ω
            hessian_inv = lx.PyTreeLinearOperator(
                (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,
                output_structure=jax.eval_shape(lambda: prev_grad),
                tags=lx.positive_semidefinite_tag,
            )
            return None, hessian_inv
        else:
            # Hessian
            assert hessian_inv is None
            mvp = hessian.mv(y_diff)
            mvp_inner = tree_dot(y_diff, mvp)
            diff_outer = _outer(grad_diff, grad_diff)
            mvp_outer = _outer(grad_diff, mvp)
            term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
            term2 = ((_outer(mvp, grad_diff) ** ω + mvp_outer**ω) / inner).ω
            new_hessian = lx.PyTreeLinearOperator(
                (hessian.pytree**ω + term1**ω - term2**ω).ω,
                output_structure=jax.eval_shape(lambda: prev_grad),
                tags=lx.positive_semidefinite_tag,
            )
            return new_hessian, None

    def no_update(hessian, hessian_inv):
        return hessian, hessian_inv

    # In particular inner = 0 on the first step (as then state.grad=0), and so for
    # this we jump straight to the line search.
    # Likewise we get inner <= eps on convergence, and so again we make no update
    # to avoid a division by zero.
    inner_nonzero = inner > jnp.finfo(inner.dtype).eps
    hessian, hessian_inv = filter_cond(
        inner_nonzero, dfp_update, no_update, hessian, hessian_inv
    )
    if hessian is None:
        return FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian_inv)
    else:
        return FunctionInfo.EvalGradHessian(f_eval, grad, hessian)


# no idea why the optimistix import is not working...
def lin_to_grad(lin_fn, y_eval, autodiff_mode=None):
    # Only the shape and dtype of y_eval is evaluated, not the value itself. (lin_fn
    # was linearized at y_eval, and the values were stored.)
    # We convert to grad after linearising for efficiency:
    # https://github.com/patrick-kidger/optimistix/issues/89#issuecomment-2447669714
    if autodiff_mode == "bwd":
        (grad,) = jax.linear_transpose(lin_fn, y_eval)(1.0)  # (1.0 is a scaling factor)
        return grad
    if autodiff_mode == "fwd":
        return jax.jacfwd(lin_fn)(y_eval)
    else:
        raise ValueError(
            "Only `autodiff_mode='fwd'` or `autodiff_mode='bwd'` are valid."
        )


_Hessian = tp.TypeVar(
    "_Hessian", FunctionInfo.EvalGradHessian, FunctionInfo.EvalGradHessianInv
)


class _MigradState(
    eqx.Module, tp.Generic[Y, Aux, SearchState, DescentState, _Hessian], strict=True
):
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    edm_max: Scalar
    search_state: SearchState
    # Updated after each descent step
    f_info: _Hessian
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS
    # Used in compat.py
    num_accepted_steps: Int[Array, ""]


class AbstractMigrad(
    AbstractMinimiser[Y, Aux, _MigradState], tp.Generic[Y, Aux, _Hessian], strict=True
):
    """Migrad algorithm"""

    rtol: eqx.AbstractVar[float]
    atol: eqx.AbstractVar[float]
    norm: eqx.AbstractVar[tp.Callable[[PyTree], Scalar]]
    use_inverse: eqx.AbstractVar[bool]
    descent: eqx.AbstractVar[AbstractDescent[Y, _Hessian, tp.Any]]
    search: eqx.AbstractVar[AbstractSearch[Y, _Hessian, FunctionInfo.Eval, tp.Any]]
    verbose: eqx.AbstractVar[frozenset[str]]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, tp.Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _MigradState:
        del options, tags
        f = tree_full_like(f_struct, 0)
        grad = tree_full_like(y, 0)
        # grad = eqx.filter_grad(lambda _y: fn(_y, args)[0])(y)
        if self.use_inverse:
            hessian_inv = _identity_pytree(y)
            # hessian_inv = _init_hessian_inv(fn, y, args)
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, hessian_inv)
        else:
            hessian = _identity_pytree(y)
            # hessian = _init_hessian(fn, y, args)
            f_info = FunctionInfo.EvalGradHessian(f, grad, hessian)
        f_info_struct = eqx.filter_eval_shape(lambda: f_info)
        return _MigradState(
            first_step=jnp.array(True),
            y_eval=y,
            edm_max=jnp.array(
                0.002 * self.rtol * 1
            ),  # https://scikit-hep.org/iminuit/reference.html#iminuit.Minuit.tol
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_accepted_steps=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, tp.Any],
        state: _MigradState,
        tags: frozenset[object],
    ) -> tuple[Y, _MigradState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)
            y_diff = (state.y_eval**ω - y**ω).ω
            if self.use_inverse:
                hessian = None
                hessian_inv = state.f_info.hessian_inv
            else:
                hessian = state.f_info.hessian
                hessian_inv = None
            f_eval_info = _dfp_update(
                f_eval, grad, state.f_info.grad, hessian, hessian_inv, y_diff
            )
            descent_state = self.descent.query(
                state.y_eval,
                f_eval_info,  # pyright: ignore
                descent_state,
            )
            f_diff = (f_eval**ω - state.f_info.f**ω).ω

            # Termination criterion
            # Since we have access to the inverse hessian,
            # we can calculate the edm
            if self.use_inverse:
                # somehow we get some outliers for bestfit if we use edm as distance measure. Is this correctly implemented?
                # cauchy_termination seems to work better...

                edm = _edm(hessian, hessian_inv, grad)
                # # https://scikit-hep.org/iminuit/reference.html#iminuit.Minuit.tol
                # terminate = edm < state.edm_max

                terminate = cauchy_termination(
                    self.rtol,
                    self.atol,
                    self.norm,
                    state.y_eval,
                    y_diff,
                    f_eval,
                    f_diff,
                )
            else:
                # TODO: can we do the EDM calculation without the inverse hessian? (or how to invert?)
                edm = state.edm_max
                # since we don't have the inverse hessian, we can't calculate the edm
                # and thus fall back to using cauchy_termination
                terminate = cauchy_termination(
                    self.rtol,
                    self.atol,
                    self.norm,
                    state.y_eval,
                    y_diff,
                    f_eval,
                    f_diff,
                )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate, edm

        def rejected(descent_state):
            return (
                y,
                state.f_info,
                state.aux,
                descent_state,
                jnp.array(False),
                state.edm_max,
            )

        y, f_info, aux, descent_state, terminate, edm = filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        if len(self.verbose) > 0:
            verbose_loss = "loss" in self.verbose
            verbose_step_size = "step_size" in self.verbose
            verbose_y = "y" in self.verbose
            verbose_edm = "edm" in self.verbose
            loss_eval = f_eval
            loss = state.f_info.f
            verbose_print(
                (verbose_loss, "Loss on this step", loss_eval),
                (verbose_loss, "Loss on the last accepted step", loss),
                (verbose_edm, "EDM on this step", edm),
                (verbose_step_size, "Step size", step_size),
                (verbose_y, "y", state.y_eval),
                (verbose_y, "y on the last accepted step", y),
            )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω
        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        state = _MigradState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            edm_max=state.edm_max,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
        )
        return y, state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, tp.Any],
        state: _MigradState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, tp.Any],
        state: _MigradState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, tp.Any]]:
        return y, aux, {}


class Migrad(AbstractMigrad[Y, Aux, _Hessian], strict=True):
    """MIGRAD minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: tp.Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: BacktrackingArmijo
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: tp.Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
        self.search = BacktrackingArmijo()
        self.verbose = verbose


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    def rosenbrock(params, args=None):
        """
        Rosenbrock function. Minimum: f(1, 1) = 0.

        https://en.wikipedia.org/wiki/Rosenbrock_function
        """
        return (1 - params["x"]) ** 2 + 100 * (params["y"] - params["x"] ** 2) ** 2

    init_params = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
    solver = Migrad(
        rtol=1e-3,
        atol=1e-5,
        use_inverse=False,
        verbose=frozenset({"loss"}),
    )

    res = optx.minimise(
        rosenbrock,
        solver,
        init_params,
        has_aux=False,
        args=None,
        options={},
        max_steps=1000,
        throw=False,
    )
    print("\nFitted params:", res.value)

    # Output:
    # Loss on this step: 1.0, Loss on the last accepted step: 0.0
    # Loss on this step: 1601.0, Loss on the last accepted step: 1.0
    # Loss on this step: 100.0, Loss on the last accepted step: 1.0
    # Loss on this step: 6.5, Loss on the last accepted step: 1.0
    # Loss on this step: 0.953125, Loss on the last accepted step: 1.0
    # Loss on this step: 0.7900390625, Loss on the last accepted step: 1.0
    # Loss on this step: 1290.612443364238, Loss on the last accepted step: 0.7900390625
    # Loss on this step: 77.57629751585345, Loss on the last accepted step: 0.7900390625
    # Loss on this step: 4.887994681492138, Loss on the last accepted step: 0.7900390625
    # Loss on this step: 0.7502141266596495, Loss on the last accepted step: 0.7900390625
    # Loss on this step: 0.6221736695061825, Loss on the last accepted step: 0.7900390625
    # Loss on this step: 0.3893128383388806, Loss on the last accepted step: 0.6221736695061825
    # Loss on this step: 0.6890522712028336, Loss on the last accepted step: 0.3893128383388806
    # Loss on this step: 0.37099240650653165, Loss on the last accepted step: 0.3893128383388806
    # Loss on this step: 0.2893730009847001, Loss on the last accepted step: 0.37099240650653165
    # Loss on this step: 0.23237012022481404, Loss on the last accepted step: 0.2893730009847001
    # Loss on this step: 0.14567474693844784, Loss on the last accepted step: 0.23237012022481404
    # Loss on this step: 0.12820002781280046, Loss on the last accepted step: 0.14567474693844784
    # Loss on this step: 0.1105530115462547, Loss on the last accepted step: 0.12820002781280046
    # Loss on this step: 0.0767684706769693, Loss on the last accepted step: 0.1105530115462547
    # Loss on this step: 0.041824852432252035, Loss on the last accepted step: 0.0767684706769693
    # Loss on this step: 0.04008144957070741, Loss on the last accepted step: 0.041824852432252035
    # Loss on this step: 0.03242225331598508, Loss on the last accepted step: 0.041824852432252035
    # Loss on this step: 0.02923669942298027, Loss on the last accepted step: 0.03242225331598508
    # Loss on this step: 0.011082169252583535, Loss on the last accepted step: 0.02923669942298027
    # Loss on this step: 0.005139586112812114, Loss on the last accepted step: 0.011082169252583535
    # Loss on this step: 0.0014244980743247939, Loss on the last accepted step: 0.005139586112812114
    # Loss on this step: 0.00027379303786038327, Loss on the last accepted step: 0.0014244980743247939
    # Loss on this step: 5.642814322914561e-05, Loss on the last accepted step: 0.00027379303786038327
    # Loss on this step: 2.525443822967524e-05, Loss on the last accepted step: 5.642814322914561e-05
    # Loss on this step: 1.990342670225858e-07, Loss on the last accepted step: 2.525443822967524e-05
    # Loss on this step: 2.373142286943067e-09, Loss on the last accepted step: 1.990342670225858e-07

    # Fitted params: {'x': Array(0.99995305, dtype=float64), 'y': Array(0.99990481, dtype=float64)}
