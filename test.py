# https://github.com/nsmith-/jaxfit/blob/roofit/tests/test_minimizer.py

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
import optimistix as optx

import iminuit

from migrad_optimistix import Migrad


jax.config.update("jax_enable_x64", True)

TOLERANCE = 1e-3


# test functions and start params from https://github.com/scikit-hep/iminuit/blob/f8e72b5146642dade56613f4e779379986354928/tests/test_functions.py#L16
def rosenbrock_fun_grad_params():
    def rosenbrock(x, y, args=None):
        """
        Rosenbrock function. Minimum: f(1, 1) = 0.

        https://en.wikipedia.org/wiki/Rosenbrock_function
        """
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    # for iminuit
    def rosenbrock_grad(x, y):
        """Gradient of Rosenbrock function."""
        return (-400 * x * (-(x**2) + y) + 2 * x - 2, -200 * x**2 + 200 * y)

    return rosenbrock, rosenbrock_grad, {"x": jnp.array(0.0), "y": jnp.array(0.0)}


def ackley_fun_grad_params():
    def ackley(x, y, args=None):
        """
        Ackley function. Minimum: f(0, 0) = 0.

        https://en.wikipedia.org/wiki/Ackley_function
        """
        term1 = -20 * jnp.exp(-0.2 * jnp.sqrt(0.5 * (x**2 + y**2)))
        term2 = -jnp.exp(0.5 * (jnp.cos(2 * jnp.pi * x) + jnp.cos(2 * jnp.pi * y)))
        return term1 + term2 + 20 + jnp.e

    return ackley, None, {"x": jnp.array(0.3), "y": jnp.array(0.2)}


def beale_fun_grad_params():
    def beale(x, y, args=None):
        """
        Beale function. Minimum: f(3, 0.5) = 0.

        https://en.wikipedia.org/wiki/Test_functions_for_optimization
        """
        term1 = 1.5 - x + x * y
        term2 = 2.25 - x + x * y**2
        term3 = 2.625 - x + x * y**3
        return term1 * term1 + term2 * term2 + term3 * term3

    return beale, None, {"x": jnp.array(0.5), "y": jnp.array(0.25)}


def matyas_fun_grad_params():
    def matyas(x, y, args=None):
        """
        Matyas function. Minimum: f(0, 0) = 0.

        https://en.wikipedia.org/wiki/Test_functions_for_optimization
        """
        return 0.26 * (x**2 + y**2) - 0.48 * x * y

    return matyas, None, {"x": jnp.array(0.5), "y": jnp.array(0.5)}


def timeit(fun):
    tic = time.monotonic()
    x = fun()
    toc = time.monotonic()
    return toc - tic, x


class iminuit_benchmark:
    def __init__(self, fun, grad, x0):
        # convert jax arrays to floats
        x0 = jax.tree.map(lambda x: x.item(), x0)

        # closure to avoid passing args
        self.fun = jax.jit(lambda x, y: fun(x, y, args=None))
        if grad is not None:
            grad = jax.jit(grad)
        self.minuit = iminuit.Minuit(self.fun, grad=grad, **x0)
        self.minuit.strategy = 1
        self.minuit.print_level = 0
        self.minuit.tol = TOLERANCE

    def benchmark(self, n=1):
        def _run_and_reset():
            self.minuit.migrad(
                ncall=1000, iterate=1, use_simplex=False
            )  # False, because we haven't implemented simplex in optimistix
            v = jnp.array(self.minuit.values)
            self.minuit.reset()
            return v

        out = []
        for _ in range(n):
            time, xmin = timeit(_run_and_reset)
            out.append(
                {"time": time, "xmin": xmin, "fmin": self.fun(x=xmin[0], y=xmin[1])}
            )

        out = pd.DataFrame(out)
        out.index.name = "run"
        return out


class optimistix_benchmark:
    def __init__(self, fun, grad, x0):
        del grad

        self.solver = Migrad(
            rtol=TOLERANCE,
            atol=TOLERANCE * 1e-2,  # ignored when using edm
            use_inverse=True,
            # verbose=frozenset({"edm"}),
        )
        # closure to pass params as pytree, p={"x": ..., "y": ...}
        self.fun = eqx.filter_jit(lambda p, args: fun(p["x"], p["y"], args))
        self.x0 = x0

    def benchmark(self, n=1):
        def _run_and_reset():
            v = optx.minimise(
                self.fun,
                self.solver,
                self.x0,
                has_aux=False,
                args=None,
                options={},
                max_steps=1000,
                throw=False,
            ).value
            return v

            # If you want to run each step by hand uncomment this block:
            # args = None
            # options = {}
            # f_struct = jax.ShapeDtypeStruct((), jnp.float64)
            # aux_struct = {}
            # tags = frozenset()

            # # Step and terminate functions.
            # # step = eqx.filter_jit(
            # #     eqx.Partial(self.solver.step, fn=self.fun, args=args, options=options, tags=tags)
            # # )
            # # terminate = eqx.filter_jit(
            # #     eqx.Partial(self.solver.terminate, fn=self.fun, args=args, options=options, tags=tags)
            # # )
            # #
            # step = eqx.Partial(self.solver.step, fn=self.fun, args=args, options=options, tags=tags)
            # terminate = eqx.Partial(self.solver.terminate, fn=self.fun, args=args, options=options, tags=tags)

            # y = self.x0
            # # Initial state before we start solving.
            # state = self.solver.init(self.fun, y, args, options, f_struct, aux_struct, tags)
            # done, result = terminate(y=y, state=state)

            # # Alright, enough setup. Let's do the solve!
            # while not done:
            #     print(f"Evaluating point {y} with value {self.fun(y, args)}.")
            #     y, state, aux = step(y=y, state=state)
            #     done, result = terminate(y=y, state=state)
            # if result != optx.RESULTS.successful:
            #     print(f"Oh no! Got error {result}.")
            # y, _, _ = self.solver.postprocess(self.fun, y, aux, args, options, state, tags, result)
            # print(f"Found solution {y} with value {self.fun(y, args)}.")
            # return y

        out = []
        for _ in range(n):
            time, xmin = timeit(_run_and_reset)
            out.append(
                {
                    "time": time,
                    "xmin": jnp.array(list(xmin.values())),
                    "fmin": self.fun(xmin, args=None),
                }
            )

        out = pd.DataFrame(out)
        out.index.name = "run"
        return out


if __name__ == "__main__":
    results = {}

    for name, fun_grad_params in [
        ("Rosenbrock", rosenbrock_fun_grad_params),
        ("Ackley", ackley_fun_grad_params),
        ("Beale", beale_fun_grad_params),
        ("Matyas", matyas_fun_grad_params),
    ]:
        fun, grad, x0 = fun_grad_params()
        n_runs = 5
        iminuit_df = iminuit_benchmark(fun, grad, x0).benchmark(n_runs)
        optimistix_migrad_df = optimistix_benchmark(fun, grad, x0).benchmark(n_runs)
        results[name] = {
            "iminuit": iminuit_df,
            "optimistix_migrad": optimistix_migrad_df,
        }

    import wadler_lindig as wl

    wl.pprint(results)

    # Output:
    # {
    #   'Rosenbrock':
    #   {
    #     'iminuit':
    #              time                                      xmin                  fmin
    #     run
    #     0    0.048580  [0.9999539813175977, 0.9998921237800471]  2.72113612688067e-08
    #     1    0.000911  [0.9999539813175977, 0.9998921237800471]  2.72113612688067e-08
    #     2    0.000788  [0.9999539813175977, 0.9998921237800471]  2.72113612688067e-08
    #     3    0.000769  [0.9999539813175977, 0.9998921237800471]  2.72113612688067e-08
    #     4    0.000762  [0.9999539813175977, 0.9998921237800471]  2.72113612688067e-08,
    #     'optimistix_migrad':
    #              time                                     xmin                    fmin
    #     run
    #     0    0.129939  [0.9999530534475961, 0.999904808470456]  2.3731422880282913e-09
    #     1    0.000367  [0.9999530534475961, 0.999904808470456]  2.3731422880282913e-09
    #     2    0.000231  [0.9999530534475961, 0.999904808470456]  2.3731422880282913e-09
    #     3    0.000219  [0.9999530534475961, 0.999904808470456]  2.3731422880282913e-09
    #     4    0.000217  [0.9999530534475961, 0.999904808470456]  2.3731422880282913e-09
    #   },
    #   'Ackley':
    #   {
    #     'iminuit':
    #              time                                               xmin                  fmin
    #     run
    #     0    0.021626  [-3.3306690738754696e-16, 1.6616936665590816e-09]  4.69998084895451e-09
    #     1    0.000835  [-3.3306690738754696e-16, 1.6616936665590816e-09]  4.69998084895451e-09
    #     2    0.000737  [-3.3306690738754696e-16, 1.6616936665590816e-09]  4.69998084895451e-09
    #     3    0.000721  [-3.3306690738754696e-16, 1.6616936665590816e-09]  4.69998084895451e-09
    #     4    0.000711  [-3.3306690738754696e-16, 1.6616936665590816e-09]  4.69998084895451e-09,
    #     'optimistix_migrad':
    #              time                                            xmin                   fmin
    #     run
    #     0    0.100112  [3.165754080192864e-06, 9.236522534532461e-07]  9.327725411623078e-06
    #     1    0.000376  [3.165754080192864e-06, 9.236522534532461e-07]  9.327725411623078e-06
    #     2    0.000235  [3.165754080192864e-06, 9.236522534532461e-07]  9.327725411623078e-06
    #     3    0.000219  [3.165754080192864e-06, 9.236522534532461e-07]  9.327725411623078e-06
    #     4    0.000215  [3.165754080192864e-06, 9.236522534532461e-07]  9.327725411623078e-06
    #   },
    #   'Beale':
    #   {
    #     'iminuit':
    #              time                                     xmin                  fmin
    #     run
    #     0    0.018181  [2.999073055962088, 0.4999131243771139]  6.09216064590447e-07
    #     1    0.000639  [2.999073055962088, 0.4999131243771139]  6.09216064590447e-07
    #     2    0.000549  [2.999073055962088, 0.4999131243771139]  6.09216064590447e-07
    #     3    0.000536  [2.999073055962088, 0.4999131243771139]  6.09216064590447e-07
    #     4    0.000530  [2.999073055962088, 0.4999131243771139]  6.09216064590447e-07,
    #     'optimistix_migrad':
    #              time                                       xmin                    fmin
    #     run
    #     0    0.083151  [2.9996681100660108, 0.49992060954810874]  1.7833748357938437e-08
    #     1    0.000370  [2.9996681100660108, 0.49992060954810874]  1.7833748357938437e-08
    #     2    0.000228  [2.9996681100660108, 0.49992060954810874]  1.7833748357938437e-08
    #     3    0.000222  [2.9996681100660108, 0.49992060954810874]  1.7833748357938437e-08
    #     4    0.000213  [2.9996681100660108, 0.49992060954810874]  1.7833748357938437e-08
    #   },
    #   'Matyas':
    #   {
    #     'iminuit':
    #              time                                              xmin                   fmin
    #     run
    #     0    0.015668  [1.3655743202889425e-14, 1.3655743202889425e-14]  7.459172896930432e-30
    #     1    0.000398  [1.3655743202889425e-14, 1.3655743202889425e-14]  7.459172896930432e-30
    #     2    0.000291  [1.3655743202889425e-14, 1.3655743202889425e-14]  7.459172896930432e-30
    #     3    0.000264  [1.3655743202889425e-14, 1.3655743202889425e-14]  7.459172896930432e-30
    #     4    0.000262  [1.3655743202889425e-14, 1.3655743202889425e-14]  7.459172896930432e-30,
    #     'optimistix_migrad':
    #              time                                              xmin                    fmin
    #     run
    #     0    0.076988  [-7.888609052210118e-30, -7.099748146989106e-30]  2.4020838972544042e-60
    #     1    0.000376  [-7.888609052210118e-30, -7.099748146989106e-30]  2.4020838972544042e-60
    #     2    0.000215  [-7.888609052210118e-30, -7.099748146989106e-30]  2.4020838972544042e-60
    #     3    0.000201  [-7.888609052210118e-30, -7.099748146989106e-30]  2.4020838972544042e-60
    #     4    0.000202  [-7.888609052210118e-30, -7.099748146989106e-30]  2.4020838972544042e-60
    #   }
    # }
