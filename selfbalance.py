import math, time
import smbus2 as smbus
import RPi.GPIO as GPIO
I2C_BUS = 1
MPU_ADDR = 0x68
PWR_MGMT_1  = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43
ACC_SENS  = 16384.0     
GYRO_SENS = 131.0     
ALPHA = 0.94
USE_GY = True 
AIN1, AIN2, PWMA = 17, 27, 18
BIN1, BIN2, PWMB = 22, 23, 13
PWM_FREQ = 20000  
INVERT_A = False 
INVERT_B = True  
Kp = 3.0
Ki = 0.0
Kd = 0.15
DEADBAND = 12       
OUT_LIM  = 30       
SLEW_PER_SEC = 60.0
SMOOTH_BETA = 0.25  
bus = smbus.SMBus(I2C_BUS)
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)  # wake IMU
GPIO.setmode(GPIO.BCM)
GPIO.setup([AIN1, AIN2, BIN1, BIN2], GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMA, GPIO.OUT); GPIO.setup(PWMB, GPIO.OUT)
pwm_a = GPIO.PWM(PWMA, PWM_FREQ); pwm_a.start(0)
pwm_b = GPIO.PWM(PWMB, PWM_FREQ); pwm_b.start(0)
def _r16(addr):
    hi = bus.read_byte_data(MPU_ADDR, addr)
    lo = bus.read_byte_data(MPU_ADDR, addr+1)
    v = (hi << 8) | lo
    return v - 65536 if v > 32767 else v
def read_acc_g():
    ax = _r16(ACCEL_XOUT_H)     / ACC_SENS
    ay = _r16(ACCEL_XOUT_H + 2) / ACC_SENS
    az = _r16(ACCEL_XOUT_H + 4) / ACC_SENS
    return ax, ay, az
def read_gyro_dps():
    gx = _r16(GYRO_XOUT_H)      / GYRO_SENS
    gy = _r16(GYRO_XOUT_H + 2)  / GYRO_SENS
    gz = _r16(GYRO_XOUT_H + 4)  / GYRO_SENS
    return gx, gy, gz
def acc_pitch(ax, ay, az):
    return math.degrees(math.atan2(ay, az))     
def _dir_pins(channel, fwd):
    if channel == 'A':
        GPIO.output(AIN1, GPIO.HIGH if fwd else GPIO.LOW)
        GPIO.output(AIN2, GPIO.LOW  if fwd else GPIO.HIGH)
    else:
        GPIO.output(BIN1, GPIO.HIGH if fwd else GPIO.LOW)
        GPIO.output(BIN2, GPIO.LOW  if fwd else GPIO.HIGH)
def set_motor(channel, speed):  
    speed = max(-100, min(100, int(speed)))
    invert = INVERT_A if channel == 'A' else INVERT_B
    fwd = (speed >= 0)
    if invert: fwd = not fwd
    _dir_pins(channel, fwd)
    duty = abs(speed)
    (pwm_a if channel=='A' else pwm_b).ChangeDutyCycle(duty)
def stop_all():
    pwm_a.ChangeDutyCycle(0); pwm_b.ChangeDutyCycle(0)
    GPIO.output([AIN1, AIN2, BIN1, BIN2], GPIO.LOW)
_last_cmd = 0.0
_smooth_cmd = 0.0
def apply_slew(cmd, dt):
    global _last_cmd
    max_step = SLEW_PER_SEC * dt
    delta = cmd - _last_cmd
    if delta >  max_step: cmd = _last_cmd + max_step
    if delta < -max_step: cmd = _last_cmd - max_step
    _last_cmd = cmd
    return cmd
def apply_smoothing(cmd):
    global _smooth_cmd
    _smooth_cmd = (1.0 - SMOOTH_BETA) * _smooth_cmd + SMOOTH_BETA * cmd
    return _smooth_cmd
def main():
    print("Calibrating gyro… keep still ~2s")
    gx_b = gy_b = gz_b = 0.0
    N = 600
    for _ in range(N):
        gx, gy, gz = read_gyro_dps()
        gx_b += gx; gy_b += gy; gz_b += gz
        time.sleep(0.003)
    gx_b /= N; gy_b /= N; gz_b /= N
    print(f"bias gx={gx_b:.2f} gy={gy_b:.2f}")
    print("Capturing level offset… keep upright ~2s")
    samples = []
    t0 = time.time()
    while time.time() - t0 < 2.0:
        ax, ay, az = read_acc_g()
        samples.append(acc_pitch(ax, ay, az))
        time.sleep(0.01)
    level_offset = sum(samples)/len(samples)
    print(f"level_offset = {level_offset:.2f} deg")
    ax, ay, az = read_acc_g()
    angle = acc_pitch(ax, ay, az)
    integral = 0.0
    prev_err = 0.0
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt = now - last
            if dt <= 0: dt = 1e-3
            last = now
            ax, ay, az = read_acc_g()
            gx, gy, gz = read_gyro_dps()
            gx -= gx_b; gy -= gy_b
            gyro_rate = gy if USE_GY else gx
            acc_ang = acc_pitch(ax, ay, az)
            angle = ALPHA * (angle + gyro_rate * dt) + (1.0 - ALPHA) * acc_ang
            err = -(angle - level_offset)
            integral += err * dt
            integral = max(-30.0, min(30.0, integral))
            deriv = (err - prev_err) / dt
            prev_err = err
            u = Kp * err + Ki * integral + Kd * deriv
            if u >  OUT_LIM: u =  OUT_LIM
            if u < -OUT_LIM: u = -OUT_LIM
            if -DEADBAND < u < DEADBAND:
                u_cmd = 0.0
            else:
                u_cmd = apply_slew(u, dt)
                u_cmd = apply_smoothing(u_cmd)
            u_cmd = -u_cmd
            if u_cmd == 0.0:
                set_motor('A', 0); set_motor('B', 0)
            else:
                set_motor('A', u_cmd); set_motor('B', u_cmd)
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()
        pwm_a.stop(); pwm_b.stop()
        GPIO.cleanup()
if __name__ == "__main__":
    main()
