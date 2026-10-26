"""Cascaded PID quadrotor controller.

Loop structure, outermost to innermost:
position (x, y, z) into velocity (world frame: vx, vy, vz)
into attitude (roll, pitch, yaw angles) into body rates (p, q, r)
into mixer into rotor speeds  (shared, controller_base.py)

The horizontal velocity loop generates a target acceleration, which
is converted into an angle of inclination using the small-angle formula θ = a / g and
then rotated from the space coordinate system to the course coordinate system by a yaw movement.
"""

import math

import rclpy
from geometry_msgs.msg import Vector3

from .pid import PID
from .controller_base import ControllerBase, GRAVITY, clamp, wrap_angle, world_to_body_yaw


class PidNode(ControllerBase):

    def __init__(self):
        super().__init__('mixer_node')
        p = self.param

        # Time constants (s)
        tau_z = p('tau_z', 1.0)
        tau_vz = p('tau_vz', 0.25)
        ti_vz = p('ti_vz', 1.25)
        tau_pos = p('tau_pos', 4.0)
        tau_vel = p('tau_vel', 1.0)
        ti_vel = p('ti_vel', 3.0)
        tau_angle = p('tau_angle', 0.3)
        tau_yaw = p('tau_yaw', 0.6)
        tau_rate_rp = p('tau_rate_rp', 0.1)
        tau_rate_yaw = p('tau_rate_yaw', 0.3)

        # PID-internal output limits
        max_vz = p('max_vz', 2.0) # m/s
        max_rate_rp = p('max_rate_rp', 3.0) # rad/s
        max_rate_yaw = p('max_rate_yaw', 0.5) # rad/s

        # Acceleration limit follows from the tilt limit (theta = a / g), so neither limit clips the other.
        self.max_accel = GRAVITY * self.max_tilt

        def ki(kp, ti):
            return kp / ti if ti > 0.0 else 0.0

        # Altitude
        self.pid_z = PID(1.0 / tau_z, 0.0, 0.0, -max_vz, max_vz)
        kp_vz = self.mass / tau_vz
        self.pid_vz = PID(kp_vz, ki(kp_vz, ti_vz), 0.0, -self.max_thrust_extra, self.max_thrust_extra)

        # Horizontal position -> velocity setpoint
        kp_pos = 1.0 / tau_pos
        self.pid_x = PID(kp_pos, 0.0, 0.0, -self.max_vel, self.max_vel)
        self.pid_y = PID(kp_pos, 0.0, 0.0, -self.max_vel, self.max_vel)

        # Horizontal velocity -> acceleration (m/s^2, mass-independent)
        kp_vel = 1.0 / tau_vel
        self.pid_vx = PID(kp_vel, ki(kp_vel, ti_vel), 0.0, -self.max_accel, self.max_accel)
        self.pid_vy = PID(kp_vel, ki(kp_vel, ti_vel), 0.0, -self.max_accel, self.max_accel)

        # Attitude -> body rates
        kp_angle = 1.0 / tau_angle
        self.pid_roll = PID(kp_angle, 0.0, 0.0, -max_rate_rp, max_rate_rp)
        self.pid_pitch = PID(kp_angle, 0.0, 0.0, -max_rate_rp, max_rate_rp)
        self.pid_yaw = PID(1.0 / tau_yaw, 0.0, 0.0, -max_rate_yaw, max_rate_yaw)

        # Body rates -> torques
        self.pid_p = PID(self.ixx / tau_rate_rp, 0.0, 0.0, -self.max_tau_rp, self.max_tau_rp)
        self.pid_q = PID(self.iyy / tau_rate_rp, 0.0, 0.0, -self.max_tau_rp, self.max_tau_rp)
        self.pid_r = PID(self.izz / tau_rate_yaw, 0.0, 0.0, -self.max_tau_yaw, self.max_tau_yaw)

        self.all_pids = [self.pid_z, self.pid_vz, self.pid_x, self.pid_y,
                         self.pid_vx, self.pid_vy, self.pid_roll, self.pid_pitch,
                         self.pid_yaw, self.pid_p, self.pid_q, self.pid_r]

        # PID-specific debug topics
        self.dbg_vel_sp = self.create_publisher(Vector3, '/debug/vel_sp', 10)
        self.dbg_accel_des = self.create_publisher(Vector3, '/debug/accel_des', 10)
        self.dbg_angles_des = self.create_publisher(Vector3, '/debug/angles_des', 10)
        self.dbg_rate_des = self.create_publisher(Vector3, '/debug/rate_des_measured', 10)
        self.dbg_rate_err = self.create_publisher(Vector3, '/debug/rate_error', 10)

    def reset_controller(self):
        for pid in self.all_pids:
            pid.reset()

    def compute(self, s, sp):
        dt = self.dt

        # Altitude: position -> vertical velocity -> extra thrust
        vz_des = self.pid_z.update(sp.z, s.z, dt)
        thrust_extra = self.pid_vz.update(vz_des, s.vz, dt)
        tilt_comp = max(0.5, math.cos(s.roll) * math.cos(s.pitch))
        thrust = (self.mass * GRAVITY + thrust_extra) / tilt_comp

        # Position -> velocity setpoint
        vx_sp = self.pid_x.update(sp.x, s.x, dt)
        vy_sp = self.pid_y.update(sp.y, s.y, dt)
        self.dbg_vel_sp.publish(Vector3(x=vx_sp, y=vy_sp, z=vz_des))

        # Velocity -> acceleration (world)
        ax_w = self.pid_vx.update(vx_sp, s.vx, dt)
        ay_w = self.pid_vy.update(vy_sp, s.vy, dt)
        self.dbg_accel_des.publish(Vector3(x=ax_w, y=ay_w, z=0.0))

        # World -> heading frame, then small-angle relation theta = a / g:
        # +x accel needs POSITIVE pitch, +y accel needs NEGATIVE roll.
        ax_b, ay_b = world_to_body_yaw(ax_w, ay_w, s.yaw)
        pitch_sp = clamp(ax_b / GRAVITY, -self.max_tilt, self.max_tilt)
        roll_sp = clamp(-ay_b / GRAVITY, -self.max_tilt, self.max_tilt)
        self.dbg_angles_des.publish(Vector3(x=roll_sp, y=pitch_sp, z=sp.yaw))

        # Attitude -> desired body rates. Yaw error is wrapped so that
        # +179 deg vs -179 deg gives 2 deg, not 358 deg.
        p_des = self.pid_roll.update(roll_sp, s.roll, dt)
        q_des = self.pid_pitch.update(pitch_sp, s.pitch, dt)
        yaw_err = wrap_angle(sp.yaw - s.yaw)
        r_des = self.pid_yaw.update(s.yaw + yaw_err, s.yaw, dt)
        self.dbg_rate_des.publish(Vector3(x=p_des, y=q_des, z=r_des))

        # Body rates -> torques
        tau_roll = self.pid_p.update(p_des, s.p, dt)
        tau_pitch = self.pid_q.update(q_des, s.q, dt)
        tau_yaw = self.pid_r.update(r_des, s.r, dt)
        self.dbg_rate_err.publish(Vector3(
            x=self.pid_p.last_error, y=self.pid_q.last_error, z=self.pid_r.last_error))

        return thrust, tau_roll, tau_pitch, tau_yaw


def main(args=None):
    rclpy.init(args=args)
    node = PidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
