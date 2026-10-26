# quadrotor-pid-vs-lqr

Cascaded PID and LQR/LQI controllers for a quadrotor in **ROS 2 Jazzy + Gazebo Harmonic**, developed for a bachelor's thesis comparing classical and optimal control (AGH University of Science and Technology, Kraków).

Both controllers share the same interface, plant parameters, mixer and actuator limits, so their results can be compared directly on the Gazebo X3 model.

## Controllers

### Cascaded PID (`mixer_node`)

Four nested loops, outermost to innermost:

```
position (x, y, z) → velocity (world frame) → attitude (roll, pitch, yaw) → body rates (p, q, r) → mixer
```

- Gains are set through time constants: `Kp = 1/τ` (scaled by mass or inertia where needed), `Ki = Kp/Ti`.
- Integral action only on `vz` and `vx/vy`; no derivative term anywhere.
- Derivative on measurement and conditional-integration anti-windup in `pid.py`.
- Desired horizontal acceleration is turned into a tilt with `θ = a/g` and rotated into the heading frame.

### LQR / LQI (`lqr_node`)

One full-state feedback law replaces the whole cascade:

```
u = u0 − K (x − x_ref),   u0 = (m·g, 0, 0, 0)
```

- Linearised around hover, the 12-state model splits into four independent subsystems: `z`, `x–pitch`, `y–roll`, `yaw`. Four small Riccati equations are solved instead of one 12×4 problem (`lqr_design.py`).
- Weights follow Bryson's rule (`Q_ii = 1/x_max²`, `R = 1/u_max²`), tied to the same limits as the PID.
- Optional integral states on x, y, z (LQI) to remove steady-state offset, e.g. under wind.
- Anti-windup, tilt limit and thrust/torque priority keep the linear law usable far from hover.

Gains and closed-loop poles are printed at startup. You can also run the design on its own:

```bash
python3 drone_controle/lqr_design.py
```

## Package layout

```
drone_controle/
├── controller_base.py   shared node: odometry, setpoint, reset, mixer, limits, debug topics
├── pid.py               single-axis PID
├── mixer_node.py        cascaded PID controller
├── lqr_design.py        LQR/LQI gain design (SciPy)
└── lqr_node.py          LQR/LQI controller
config/
├── pid_params.yaml      plant, shared limits (/** section) and PID parameters
└── lqr_params.yaml      LQR/LQI weights
```

A new controller only has to subclass `ControllerBase` and implement `compute(state, setpoint)` and `reset_controller()`.

## Requirements

- Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (`ros-jazzy-ros-gz`)
- Python: `numpy`, `scipy`
- A Gazebo world with the X3 quadrotor (MulticopterMotorModel and OdometryPublisher plugins)

## Build

```bash
cd ~/ros2_ws/src
git clone https://github.com/StalkerAVG/quadrotor-pid-vs-lqr.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --packages-select drone_controle
source install/setup.bash
```

## Run

1. Start Gazebo with the X3 world.

2. Bridge the topics:

   ```bash
   source /opt/ros/jazzy/setup.bash
    ros2 run ros_gz_bridge parameter_bridge \
      /X3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry \
      /X3/gazebo/command/motor_speed@actuator_msgs/msg/Actuators]gz.msgs.Actuators \
      /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock
   ```

3. Start one controller:

   ```bash
   CFG=$(ros2 pkg prefix drone_controle)/share/drone_controle/config

   # cascaded PID
   ros2 run drone_controle mixer_node --ros-args --params-file $CFG/pid_params.yaml

   # LQR / LQI (reads the shared plant and limits from pid_params.yaml)
   ros2 run drone_controle lqr_node --ros-args \
     --params-file $CFG/pid_params.yaml --params-file $CFG/lqr_params.yaml
   ```

4. Send a setpoint (always a full pose; missing fields default to 0):

   ```bash
   ros2 topic pub --once /setpoint geometry_msgs/msg/Pose \
     "{position: {x: 2.0, y: 0.0, z: 2.0}, orientation: {w: 1.0}}"
   ```

## Interface

| Topic | Type | Direction |
|---|---|---|
| `/X3/odometry` | `nav_msgs/Odometry` | in |
| `/setpoint` | `geometry_msgs/Pose` | in (position + yaw) |
| `/X3/gazebo/command/motor_speed` | `actuator_msgs/Actuators` | out |
| `/debug/setpoint` | `geometry_msgs/Pose` | setpoint echoed every cycle |
| `/debug/angles`, `/debug/angles_des` | `geometry_msgs/Vector3` | measured / desired roll, pitch, yaw |
| `/debug/vel_world`, `/debug/rate_measured` | `geometry_msgs/Vector3` | world velocity, body rates |
| `/debug/torque_cmd` | `geometry_msgs/Vector3` | commanded torques after limits |
| `/debug/control_effort` | `geometry_msgs/Vector3` | `x = Σω²`, `y = Σ(ω − ω_hover)²` |
| `/debug/lqr_integral` | `geometry_msgs/Vector3` | LQI integral states |

Conventions: ENU world frame, FLU body frame, ZYX Euler angles. Controllers reset automatically when simulation time jumps backwards (Gazebo world reset).

## Parameters

Plant values (mass 1.5 kg, inertia, arm lengths, motor constants) come from the X3 `model.sdf` and live in the `/**` section of `pid_params.yaml`, so both controllers always use the same plant and limits. Change them only there.

Set `hover_only: true` to send constant hover thrust with no feedback (useful for checking the mixer).

## License

See [LICENSE](LICENSE).
