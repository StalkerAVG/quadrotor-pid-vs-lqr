"""Single-axis PID controller.

One instance per controlled quantity. A cascaded quadrotor controller
uses roughly nine of these: x, y, z position; vx, vy, vz velocity;
roll, pitch, yaw attitude; p, q, r body rates.

One detail that matters a lot:
Conditional integration (anti-windup). If the output is already
saturated, integrating further error only builds up a value that must
later be "unwound", causing large overshoot. So the integrator is
frozen whenever the output is clamped and the error would push it
further in the same direction.
"""


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, derivative_tau=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max

        # First-order low-pass time constant on the derivative term.
        # Set to a few sample periods if odometry is noisy; 0 disables it.
        self.derivative_tau = derivative_tau

        self.reset()

    def reset(self):
        """Call on arming, mode change, or after the sim is paused."""
        self.integral = 0.0
        self.prev_measurement = None
        self.derivative = 0.0

    def update(self, setpoint, measurement, dt):
        if dt <= 0.0:
            return 0.0

        error = setpoint - measurement

        p_term = self.kp * error
        
        if self.prev_measurement is None:
            raw_derivative = 0.0
        else:
            raw_derivative = -(measurement - self.prev_measurement) / dt
        self.prev_measurement = measurement

        if self.derivative_tau > 0.0:
            alpha = dt / (self.derivative_tau + dt)
            self.derivative += alpha * (raw_derivative - self.derivative)
        else:
            self.derivative = raw_derivative

        d_term = self.kd * self.derivative

        candidate_integral = self.integral + error * dt
        unclamped = p_term + self.ki * candidate_integral + d_term

        saturated_high = unclamped > self.out_max and error > 0.0
        saturated_low = unclamped < self.out_min and error < 0.0

        if not (saturated_high or saturated_low):
            self.integral = candidate_integral

        i_term = self.ki * self.integral

        output = p_term + i_term + d_term
        output = max(self.out_min, min(self.out_max, output))

        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term
        self.last_error = error

        return output
