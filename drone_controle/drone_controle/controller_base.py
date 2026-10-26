"""Shared plumbing for every controller in the PID vs LQR comparison.

Everything that is NOT the control law lives here, so all controllers get
identical inputs, outputs, limits and logging:
 - plant parameters and shared actuator limits
 - odometry subscription, teleport-glitch filter, state extraction
 - setpoint topic /setpoint and its per-cycle echo /debug/setpoint
 - automatic reset when sim time jumps backwards (Gazebo world reset)
 - the mixer: collective thrust + 3 torques -> 4 rotor speeds
 - the debug topics read by analyze_bags.py

A controller inherits/subclasses ControllerBase and implements only:
compute(state, sp) -> (thrust, tau_roll, tau_pitch, tau_yaw) 
and
reset_controller()

Conventions: ENU world frame, FLU body frame, ZYX Euler angles.
"""

import math
from dataclasses import dataclass

from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from geometry_msgs.msg import Pose, Vector3

# Constants
GRAVITY = 9.81
MAX_PLAUSIBLE_SPEED = 30.0 # m/s

# This thing was validated by tests, so without them and proof DO NOT MODIFY (╯°□°)╯︵ ┻━┻
MIXER = [
    #roll pitch yaw
    [-1, -1, -1],# rotor 0  ccw  front right
    [+1, +1, -1],# rotor 1  ccw  rear left
    [+1, -1, +1],# rotor 2  cw   front left
    [-1, +1, +1],# rotor 3  cw   rear right
]


def euler_from_quaternion(x, y, z, w):
    """ZYX (yaw-pitch-roll) Euler angles from a quaternion."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    t2 = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(t2)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw

def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def body_to_world(u, v, w, roll, pitch, yaw):
    """Rotate a body-frame vector into the world frame (ZYX convention)."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    x = (cp * cy)*u + (sr * sp * cy - cr * sy)*v + (cr * sp * cy + sr * sy) *w
    y = (cp * sy) *u + (sr * sp * sy + cr * cy) * v + (cr * sp * sy - sr * cy)*w
    z = (-sp)*u + (sr * cp)*v + (cr * cp)*w
    return x, y, z


def world_to_body_yaw(x, y, yaw):
    """Rotate a horizontal world-frame vector into the heading frame."""
    sy, cy = math.sin(yaw), math.cos(yaw)
    return cy * x + sy * y, -sy * x + cy * y


@dataclass
class State:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float # world frame
    vy: float
    vz: float
    p: float # body rates
    q: float
    r: float

@dataclass
class Setpoint:
    x: float
    y: float
    z: float
    yaw: float


class ControllerBase(Node):

    def __init__(self, node_name):
        super().__init__(node_name)
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        p = self.param
        # Plant (model.sdf)
        self.mass = p('mass', 1.5)
        self.arm_x = p('arm_x', 0.13)
        self.arm_y = p('arm_y', 0.21)
        self.k_thrust = p('motor_constant', 8.54858e-06)
        self.k_moment = p('moment_constant', 0.016)
        self.max_omega = p('max_rot_velocity', 800.0)
        self.ixx = p('ixx', 0.0347563)
        self.iyy = p('iyy', 0.07)
        self.izz = p('izz', 0.0977)

        # Limits shared by ALL controllers (fair comparison)
        self.max_vel = p('max_vel', 1.5) # m/s
        self.max_tilt = p('max_tilt', 0.30) # rad (approximately 17.2 deg)
        self.max_thrust_extra = p('max_thrust_extra', 8.0)# N around hover
        self.max_tau_rp = p('max_tau_rp', 1.0) # N*m, roll/pitch
        self.max_tau_yaw = p('max_tau_yaw', 0.1) # N*m

        self.hover_only = p('hover_only', False)

        self.sp = Setpoint(p('x_sp', 0.0), p('y_sp', 0.0),
                           p('z_sp', 2.0), p('yaw_sp', 0.0))

        self.dt = 1.0 / p('control_rate', 250.0)

        self.hover_omega = math.sqrt((self.mass * GRAVITY / 4.0) / self.k_thrust)
        self.get_logger().info(f'hover thrust {self.mass * GRAVITY:.2f} N total, '
            f'{self.hover_omega:.1f} rad/s per rotor')

        self.latest_odom = None
        self.last_good_vel_body = (0.0, 0.0, 0.0)
        self.last_sim_ns = self.get_clock().now().nanoseconds

        self.cmd_pub = self.create_publisher(
            Actuators, '/X3/gazebo/command/motor_speed', 10)
        self.create_subscription(Odometry, '/X3/odometry', self._odom_cb, 10)
        self.create_subscription(Pose, '/setpoint', self._setpoint_cb, 10)

        # Debug topics 
        self.dbg_setpoint = self.create_publisher(Pose, '/debug/setpoint', 10)
        self.dbg_angles = self.create_publisher(Vector3, '/debug/angles', 10)
        self.dbg_vel = self.create_publisher(Vector3, '/debug/vel', 10)
        self.dbg_vel_world = self.create_publisher(Vector3, '/debug/vel_world', 10)
        self.dbg_rate = self.create_publisher(Vector3, '/debug/rate_measured', 10)
        self.dbg_tau = self.create_publisher(Vector3, '/debug/torque_cmd', 10)
        self.dbg_effort = self.create_publisher(Vector3, '/debug/control_effort', 10)

        self.create_timer(self.dt, self._loop)

    def compute(self, state, sp):
        """Return (thrust [N], tau_roll, tau_pitch, tau_yaw [N*m])."""
        raise NotImplementedError

    def reset_controller(self):
        """Clear all internal state (integrators, filters)."""
        raise NotImplementedError

    def param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def mix(self, thrust, tau_roll, tau_pitch, tau_yaw):
        """Convert one collective thrust and three torques into 4 rotor speeds."""
        speeds = []
        for roll_s, pitch_s, yaw_s in MIXER:
            f = (thrust / 4.0
                 + roll_s * tau_roll / (4.0 * self.arm_y)
                 + pitch_s * tau_pitch / (4.0 * self.arm_x)
                 + yaw_s * tau_yaw / (4.0 * self.k_moment))
            f = max(0.0, f)
            omega = math.sqrt(f / self.k_thrust)
            speeds.append(min(omega, self.max_omega))
        return speeds

    def _odom_cb(self, msg):
        self.latest_odom = msg

    def _setpoint_cb(self, msg):
        self.sp.x = msg.position.x
        self.sp.y = msg.position.y
        self.sp.z = msg.position.z
        q = msg.orientation
        if q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w > 0.001:
            _, _, self.sp.yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

    def _read_state(self):
        o = self.latest_odom
        q = o.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

        u, v, w = (o.twist.twist.linear.x, o.twist.twist.linear.y,
                   o.twist.twist.linear.z)
        if max(abs(u), abs(v), abs(w)) > MAX_PLAUSIBLE_SPEED:
            u, v, w = self.last_good_vel_body
        else:
            self.last_good_vel_body = (u, v, w)

        vx, vy, vz = body_to_world(u, v, w, roll, pitch, yaw)
        ang = o.twist.twist.angular

        self.dbg_angles.publish(Vector3(x=roll, y=pitch, z=yaw))
        self.dbg_vel.publish(Vector3(x=u, y=v, z=w))
        self.dbg_vel_world.publish(Vector3(x=vx, y=vy, z=vz))
        self.dbg_rate.publish(Vector3(x=ang.x, y=ang.y, z=ang.z))

        p = o.pose.pose.position
        return State(p.x, p.y, p.z, roll, pitch, yaw, vx, vy, vz, ang.x, ang.y, ang.z)

    def _publish_setpoint_echo(self):
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = self.sp.x, self.sp.y, self.sp.z
        msg.orientation.z = math.sin(self.sp.yaw / 2.0)
        msg.orientation.w = math.cos(self.sp.yaw / 2.0)
        self.dbg_setpoint.publish(msg)

    def _publish_speeds(self, speeds):
        msg = Actuators()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = speeds
        self.cmd_pub.publish(msg)

        effort_raw = sum(w * w for w in speeds)
        effort_dev = sum((w - self.hover_omega) ** 2 for w in speeds)
        self.dbg_effort.publish(Vector3(x=effort_raw, y=effort_dev, z=0.0))

    def _loop(self):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self.last_sim_ns:
            self.get_logger().warn('Sim reset detected, clearing controller state')
            self.last_good_vel_body = (0.0, 0.0, 0.0)
            self.reset_controller()
        self.last_sim_ns = now_ns

        if self.latest_odom is None:
            return

        if self.hover_only:
            self._publish_speeds(self.mix(self.mass * GRAVITY, 0.0, 0.0, 0.0))
            return

        self._publish_setpoint_echo()
        state = self._read_state()

        thrust, tau_r, tau_p, tau_y = self.compute(state, self.sp)

        # Same actuator limits for every controller.
        tau_r = clamp(tau_r, -self.max_tau_rp, self.max_tau_rp)
        tau_p = clamp(tau_p, -self.max_tau_rp, self.max_tau_rp)
        tau_y = clamp(tau_y, -self.max_tau_yaw, self.max_tau_yaw)
        self.dbg_tau.publish(Vector3(x=tau_r, y=tau_p, z=tau_y))

        self._publish_speeds(self.mix(thrust, tau_r, tau_p, tau_y))
