"""LQR gain design for the quadrotor linearised around hover.

The 12-state hover model splits into four independent subsystems, so four
small Riccati equations are solved instead of one 12x4 problem.

Q and R follow Bryson's rule: Q_ii = 1 / x_i_max^2, R = 1 / u_max^2.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

GRAVITY = 9.81

@dataclass
class Plant:
    mass: float = 1.5
    ixx: float = 0.0347563
    iyy: float = 0.07
    izz: float = 0.0977

@dataclass
class Weights:
    """Bryson's rule: the largest acceptable value of every state and input."""
    # States
    pos_max: float = 2.0 # horizontal position error in meters
    z_max: float = 1.0 # altitude error in meters (S1 step)
    yaw_max: float = math.pi / 2.0 #yaw error in radians (S3 step)
    vel_max: float = 1.5 #max_vel in meters per second
    vz_max: float = 2.0 # max_vz of the PID in m/s
    tilt_max: float = 0.30 #max_tilt in radians per second
    rate_rp_max: float = 3.0 # max_rate_rp of the PID in rad/s
    rate_yaw_max: float = 0.5 # max_rate_yaw of the PID rad/s
    # Integral states (m*s). Smaller value = stronger integral action.
    int_xy_max: float = 5.0
    int_z_max: float = 1.0
    # Inputs
    thrust_max: float = 8.0 #max_thrust_extra in Newtons
    tau_rp_max: float = 1.0 #max_tau_rp Newtons*meter
    tau_yaw_max: float = 0.1# max_tau_yaw N*m


def bryson(*maxima):
    return np.diag([1.0 / (m * m) for m in maxima])


def chain(*gains):
    """A, B of a chain of integrators: x1' = g1*x2, x2' = g2*x3, ..., xn' = gn*u."""
    n = len(gains)
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = gains[i]
    B = np.zeros((n, 1))
    B[n - 1, 0] = gains[n - 1]
    return A, B


def with_integral(A, B):
    """Put the integral of the first state in front of the state vector."""
    n = A.shape[0]
    Aa = np.zeros((n + 1, n + 1))
    Aa[0, 1] = 1.0
    Aa[1:, 1:] = A
    Ba = np.vstack([np.zeros((1, B.shape[1])), B])
    return Aa, Ba


def controllability_rank(A, B):
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return int(np.linalg.matrix_rank(np.hstack(blocks)))


def lqr(A, B, Q, R):
    if controllability_rank(A, B) < A.shape[0]:
        raise ValueError('subsystem is not controllable')
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    poles = np.linalg.eigvals(A - B @ K)
    if np.any(poles.real >= 0.0):
        raise ValueError('closed loop is not stable')
    return K[0], poles


def design(plant=None, w=None, integral=True):
    """Gains for the four subsystems.

    Returns {name: {'K': array, 'poles': array, 'states': [names]}}.
    State order is exactly the order the node must use when it builds the
    state vector: [integral (if enabled), position, velocity, angle, rate].
    """
    plant = plant or Plant()
    w = w or Weights()

    subsystems = {
        'z': (chain(1.0, 1.0 / plant.mass),
              [w.z_max, w.vz_max], w.int_z_max, w.thrust_max,
              ['z', 'vz']),
        'x': (chain(1.0, GRAVITY, 1.0, 1.0 / plant.iyy),
              [w.pos_max, w.vel_max, w.tilt_max, w.rate_rp_max],
              w.int_xy_max, w.tau_rp_max,
              ['x', 'vx', 'pitch', 'q']),
        'y': (chain(1.0, -GRAVITY, 1.0, 1.0 / plant.ixx),
              [w.pos_max, w.vel_max, w.tilt_max, w.rate_rp_max],
              w.int_xy_max, w.tau_rp_max,
              ['y', 'vy', 'roll', 'p']),
        'yaw': (chain(1.0, 1.0 / plant.izz),
                [w.yaw_max, w.rate_yaw_max], None, w.tau_yaw_max,
                ['yaw', 'r']),
    }

    out = {}
    for name, ((A, B), x_max, int_max, u_max, states) in subsystems.items():
        if integral and int_max is not None:
            A, B = with_integral(A, B)
            x_max = [int_max] + x_max
            states = ['int_' + states[0]] + states
        K, poles = lqr(A, B, bryson(*x_max), bryson(u_max))
        out[name] = {'K': K, 'poles': poles, 'states': states}
    return out


def format_gains(gains, control_rate=250.0):
    lines = []
    fastest = 0.0
    for name, g in gains.items():
        lines.append(f'--- {name} ---')
        for s, k in zip(g['states'], g['K']):
            lines.append(f'  K[{s:<6}] = {k: .5g}')
        poles = sorted(g['poles'], key=lambda p: (p.real, p.imag))
        lines.append('  poles: ' + ', '.join(
            f'{p.real:.3g}{p.imag:+.3g}j' if abs(p.imag) > 1e-9 else f'{p.real:.3g}'
            for p in poles))
        fastest = max(fastest, max(abs(p) for p in poles))
    lines.append(f'fastest pole {fastest:.1f} rad/s at {control_rate:.0f} Hz: '
                 f'|pole| * dt = {fastest / control_rate:.3f} '
                 f'(continuous design is valid while this is << 1)')
    return '\n'.join(lines)


if __name__ == '__main__':
    print('-LQI (integral on x, y, z)-')
    print(format_gains(design(integral=True)))
    print()
    print('-plain LQR-')
    print(format_gains(design(integral=False)))
