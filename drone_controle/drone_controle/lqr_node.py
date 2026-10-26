"""LQR quadrotor controller (full-state feedback, optional integral action).

One state-feedback law replaces the whole PID cascade:

    u = u0 - K * (x - x_ref), u0 = (m * g, 0, 0, 0)

The gains come from lqr_design.py: the hover model splits into four
independent subsystems (z, x-pitch, y-roll, yaw), each with its own K.
"""

import math
from dataclasses import fields

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3

from .controller_base import MIXER, ControllerBase, GRAVITY, clamp, wrap_angle, world_to_body_yaw
from .lqr_design import Plant, Weights, design, format_gains


class LqrNode(ControllerBase):

    def __init__(self):
        super().__init__('lqr_node')
        p = self.param

        # true = LQI (integral of the x, y, z position error), false = plain LQR
        self.use_integral = p('use_integral', True)

        # Largest position error that is fed to the integrators (0 = no limit).
        self.int_err_limit_xy = p('int_err_limit_xy', 0.25) # m
        self.int_err_limit_z = p('int_err_limit_z', 0.10) # m

        # Largest horizontal position error that is fed to the control law (0 = no limit).
        self.pos_err_limit = p('pos_err_limit', 2.0) # m

        # Bryson weights
        weights = Weights(**{f.name: p('w_' + f.name, f.default) for f in fields(Weights)})
        plant = Plant(self.mass, self.ixx, self.iyy, self.izz)

        gains = design(plant, weights, self.use_integral)
        self.k_z = gains['z']['K']
        self.k_x = gains['x']['K']
        self.k_y = gains['y']['K']
        self.k_yaw = gains['yaw']['K']
        self.get_logger().info(
            ('LQI' if self.use_integral else 'LQR') + ' gains:\n'
            + format_gains(gains, 1.0 / self.dt))

        # Integrals of the position error, kept in the WORLD frame: they
        # represent a world-fixed disturbance (wind), so they must not turn together with the drone when it yaws.
        self.int_x = 0.0
        self.int_y = 0.0
        self.int_z = 0.0

        self.dbg_integral = self.create_publisher(Vector3, '/debug/lqr_integral', 10)
        self.dbg_angles_des = self.create_publisher(Vector3, '/debug/angles_des', 10)

    def reset_controller(self):
        self.int_x = 0.0
        self.int_y = 0.0
        self.int_z = 0.0

    @staticmethod
    def _limited(value, limit):
        return clamp(value, -limit, limit) if limit > 0.0 else value

    @staticmethod
    def _limited_xy(x, y, limit):
        """Shorten a horizontal vector to `limit`, keeping its direction."""
        length = math.hypot(x, y)
        if limit <= 0.0 or length <= limit:
            return x, y
        return x * limit / length, y * limit / length

    def _law(self, k, integral, *state):
        """u = -K x. With plain LQR the integral state is not part of K."""
        x = (integral, *state) if self.use_integral else state
        return -float(np.dot(k, x))

    def compute(self, s, sp):
        dt = self.dt

        # State error (the reference for everything except position and yaw is 0).
        ex_w = s.x - sp.x
        ey_w = s.y - sp.y
        ez = s.z - sp.z
        eyaw = wrap_angle(s.yaw - sp.yaw)

        ex, ey = world_to_body_yaw(
            *self._limited_xy(ex_w, ey_w, self.pos_err_limit), s.yaw)
        vx, vy = world_to_body_yaw(s.vx, s.vy, s.yaw)

        def outputs(int_x, int_y, int_z):
            ix, iy = world_to_body_yaw(int_x, int_y, s.yaw)
            return (self._law(self.k_z, int_z, ez, s.vz),
                    self._law(self.k_y, iy, ey, vy, s.roll, s.p),
                    self._law(self.k_x, ix, ex, vx, s.pitch, s.q))

        if self.use_integral:
            thrust_now, roll_now, pitch_now = outputs(self.int_x, self.int_y, self.int_z)

            cand_z = self.int_z + self._limited(ez, self.int_err_limit_z) * dt
            thrust_new, _, _ = outputs(self.int_x, self.int_y, cand_z)
            if not (abs(thrust_new) > self.max_thrust_extra
                    and abs(thrust_new) > abs(thrust_now)):
                self.int_z = cand_z

            dix, diy = self._limited_xy(ex_w, ey_w, self.int_err_limit_xy)
            cand_x = self.int_x + dix * dt
            cand_y = self.int_y + diy * dt
            _, roll_new, pitch_new = outputs(cand_x, cand_y, self.int_z)
            deeper = ((abs(roll_new) > self.max_tau_rp and abs(roll_new) > abs(roll_now))
                      or (abs(pitch_new) > self.max_tau_rp and abs(pitch_new) > abs(pitch_now)))
            if not deeper:
                self.int_x, self.int_y = cand_x, cand_y

            self.dbg_integral.publish(Vector3(x=self.int_x, y=self.int_y, z=self.int_z))

        thrust_extra, tau_roll, tau_pitch = outputs(self.int_x, self.int_y, self.int_z)
        tau_yaw = -float(np.dot(self.k_yaw, (eyaw, s.r)))

        k_roll, k_p = self.k_y[-2], self.k_y[-1]
        k_pitch, k_q = self.k_x[-2], self.k_x[-1]
        roll_des = (tau_roll + k_roll * s.roll + k_p * s.p) / k_roll
        pitch_des = (tau_pitch + k_pitch * s.pitch + k_q * s.q) / k_pitch
        roll_des = clamp(roll_des, -self.max_tilt, self.max_tilt)
        pitch_des = clamp(pitch_des, -self.max_tilt, self.max_tilt)
        tau_roll = k_roll * (roll_des - s.roll) - k_p * s.p
        tau_pitch = k_pitch * (pitch_des - s.pitch) - k_q * s.q
        self.dbg_angles_des.publish(Vector3(x=roll_des, y=pitch_des, z=sp.yaw))

        tau_roll = clamp(tau_roll, -self.max_tau_rp, self.max_tau_rp)
        tau_pitch = clamp(tau_pitch, -self.max_tau_rp, self.max_tau_rp)
        tau_yaw = clamp(tau_yaw, -self.max_tau_yaw, self.max_tau_yaw)

        # u0 = m * g is the equilibrium input of the linearisation.
        thrust_extra = clamp(thrust_extra, -self.max_thrust_extra, self.max_thrust_extra)
        thrust = self.mass * GRAVITY + thrust_extra

        # Torque has priority over thrust: keep the collective thrust in the
        # range where the mixer can still produce the requested torques.
        lo, hi = self._thrust_range(tau_roll, tau_pitch, tau_yaw)
        thrust = clamp(thrust, lo, hi)

        return thrust, tau_roll, tau_pitch, tau_yaw

    def _thrust_range(self, tau_roll, tau_pitch, tau_yaw):
        """Collective thrust for which no rotor leaves [0, max force].

        mix() gives every rotor thrust / 4 plus its share of the torques. If
        thrust is so high that rotors hit max_rot_velocity, the torque share
        is cut off and the drone cannot turn itself level any more.
        """
        shares = [roll_s * tau_roll / (4.0 * self.arm_y)
                  + pitch_s * tau_pitch / (4.0 * self.arm_x)
                  + yaw_s * tau_yaw / (4.0 * self.k_moment)
                  for roll_s, pitch_s, yaw_s in MIXER]
        f_max = self.k_thrust * self.max_omega ** 2
        hi = 4.0 * (f_max - max(shares))
        lo = min(4.0 * -min(shares), hi)
        return lo, hi


def main(args=None):
    rclpy.init(args=args)
    node = LqrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
