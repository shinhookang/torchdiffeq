from .solvers import FixedGridODESolver
from . import rk_common
import torch
from .misc import _scaled_dot_product, _convert_to_tensor


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in func(t, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2

    
class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4
    
class Dopri5_fixed(FixedGridODESolver):
    
    def step_func(self, func, t, dt, y):
        import collections

        _ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha beta c_sol c_error')

        tableau = _ButcherTableau(
        alpha=[1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.],
        beta=[
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ],
        c_sol=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        c_error=[
            35 / 384 - 1951 / 21600,
            0,
            500 / 1113 - 22642 / 50085,
            125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400,
            11 / 84 - 649 / 6300,
            -1. / 60.,
        ],
        )
        
        f0 = tuple( f_ for f_ in func(t,y) )
        t0 = t
        y0 = y
        stage = 1
        k = tuple(map(lambda x: [x], f0))
        for alpha_i, beta_i in zip(tableau.alpha, tableau.beta):
            ti = t0 + alpha_i * dt
            if stage < 6:
                yi = tuple( y0_ + _scaled_dot_product(dt, beta_i, k_) for y0_, k_ in zip(y0, k))
                tuple(k_.append(f_) for k_, f_ in zip(k, func(ti, yi)))
            else:
                out = tuple(_scaled_dot_product(dt,beta_i,k_) for k_ in k)
            stage = stage + 1

        return out

    @property
    def order(self):
        return 4
        
