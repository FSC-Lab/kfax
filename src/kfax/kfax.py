"""
Copyright © 2024 Hs293Go

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import functools
import operator
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from jax import Array


class KalmanOperatingPoint(NamedTuple):
    x_op: Array
    kf_cov: Array


class KalmanFilter:
    """Implements the Extended Kalman Filter, optionally using jax autodiff to linearize
    the state dynamics and observation models
    """

    def __init__(
        self,
        fcn: Callable[[Array, Array, Array], Array],
        hfcn: Callable[[Array, Any], Array],
        in_cov: Array,
        obs_cov: Array,
        fjac: Optional[Callable[[Array, Array, Array], Tuple[Array, Array]]] = None,
        hjac: Optional[Callable[[Array, Any], Array]] = None,
        resid_fcn: Callable[[Array, Array], Array] = operator.sub,
        corr_fcn: Callable[[Array, Array], Array] = operator.add,
    ):
        """Initializes the functional components and constant parameters of this
        Extended Kalman Filter

        Parameters
        ----------
        fcn : Callable[[Array, Array, Array], Array]
            The discrete-time state dynamics model.
            Must have a signature f(x, u, dt) -> dx
        hfcn : Callable[[Array], Array]
            The observation model.
            Must have a signature h(x, args...) -> dx where args are any number of
            additional runtime parameters to be passed to the observation model
        in_cov : Array
            The process covariance matrix (Q). Must be square and diagonal
        obs_cov : Array
            The observation covariance matrix (R). Must be square and diagonal
        fjac : Optional[Callable[[Array, Array, Array], Tuple[Array, Array]]], optional
            An user-defined function to compute the linearized system AND input matrix.
            If provided, it must have signature (df/d{x,u})(x, u, dt) -> F, G;
            If left unspecified, jax.jacfwd will be used to automatically evaluate the
            matrices, by default None
        hjac : Optional[Callable[[Array], Array]], optional
            An user-defined function to compute the linearized observation matrix.
            If provided, it must have sigature (dh/dx)(x, args...) -> H;
            If left unspecified, jax.jacfwd will be used to automatically evaluate the
            matrix, by default None
        resid_fcn : Callable[[Array, Array], Array], optional
            An user-defined function to compute the innovation if simple vector
            subtraction is insufficient, e.g. if the result must be projected to some
            manifold, by default operator.sub
        corr_fcn : Callable[[Array, Array], Array], optional
            An user-defined function to apply corrections to the state if simple vector
            addition is sufficient, by default operator.add
        """
        self._fcn = fcn
        self._hfcn = hfcn
        self._fjac = (
            fjac if fjac is not None else jax.jit(jax.jacfwd(fcn, argnums=(0, 1)))
        )
        self._hjac = hjac if hjac is not None else jax.jit(jax.jacfwd(hfcn, argnums=0))
        self._in_cov = in_cov
        self._obs_cov = obs_cov
        self._resid_fcn = resid_fcn
        self._corr_fcn = corr_fcn

    @property
    def in_cov(self):
        return self._in_cov

    @property
    def obs_cov(self):
        return self._obs_cov

    @functools.partial(jax.jit, static_argnames=["self"], donate_argnums=[1, 2])
    def predict(
        self, x_op: Array, kf_cov: Array, u: Array, dt: Array
    ) -> KalmanOperatingPoint:
        """Evaluates EKF prediction / Time Update

        Parameters
        ----------
        x_op : Array
            The current state estimate
        kf_cov : Array
            The current EKF covariance matrix
        u : Array
            The system inputs at this timestep
        dt : Array
            The time interval for the update

        Returns
        -------
        KalmanOperatingPoint
            A NamedTuple containing the a-priori state estimate (prediction) and EKF
            covariance matrix
        """

        fjac, gjac = self._fjac(x_op, u, dt)
        kf_cov = la.multi_dot((fjac, kf_cov, fjac.T)) + la.multi_dot(
            (gjac, self.in_cov, gjac.T)
        )
        x_op = self._fcn(x_op, u, dt)
        return KalmanOperatingPoint(x_op, kf_cov)

    @functools.partial(jax.jit, static_argnames=["self"], donate_argnums=[1, 2])
    def update(
        self, x_op: Array, kf_cov: Array, y: Array, args: Tuple[Array, ...] = ()
    ) -> KalmanOperatingPoint:
        """Evaluates EKF update / Measurement Update.

        Details
        -------
        This function is able to carry out a batch of updates at once by leveraging
        jax.vmap. In this circumstance, a series of observations stacked along the first
        axis (row-wise) is expected. The operating point will be broadcasted to be
        vmapped alongside the observations. More subtly, this function also assumes the
        additional arguments are already in stacked form, compatible for vmap alongside
        the observations. This is intended to account for distinct runtime parameters
        corresponding to each observation in the batch. If this is not the use-case,
        users may need to broadcast their additional arguments.

        Parameters
        ----------
        x_op : Array
            The current state estimate
        kf_cov : Array
            The current EKF covariance matrix
        y : Array
            The current exteroceptive observation (sensor) data
        args : Tuple[Array, ...], optional
            Any number of additional parameters to be passed to the observation model,
            by default ()

        Returns
        -------
        KalmanOperatingPoint
            A NamedTuple containing the a-priori state estimate (prediction) and EKF
            covariance matrix
        """

        if y.size == 0:
            return KalmanOperatingPoint(x_op, kf_cov)

        y_wa = jnp.atleast_2d(y)
        x_wa = jnp.atleast_2d(x_op)
        n_meas = y_wa.shape[0]
        shape = (n_meas, x_wa.shape[1])
        x_wa = jnp.broadcast_to(x_wa, shape)

        hjac = jax.vmap(self._hjac)(x_wa, *args)
        hjac = hjac.reshape(-1, hjac.shape[-1])

        hx = jax.vmap(self._hfcn)(x_wa, *args)
        obs_cov = jnp.kron(jnp.eye(n_meas), self.obs_cov)

        resid_cov = hjac @ kf_cov @ hjac.T + obs_cov
        kf_gain = la.solve(resid_cov.T, hjac @ kf_cov).T
        kf_cov -= (
            kf_gain @ hjac @ kf_cov
            + kf_cov @ hjac.T @ kf_gain.T
            - kf_gain @ resid_cov @ kf_gain.T
        )

        z = self._resid_fcn(y_wa, hx).ravel()
        x_op = self._corr_fcn(x_op, kf_gain @ z)
        return KalmanOperatingPoint(x_op, kf_cov)
