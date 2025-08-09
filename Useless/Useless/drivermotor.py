import RPi.GPIO as GPIO
import time

# Set up GPIO pins
AIN1 = 17
AIN2 = 27
PWMA = 18

GPIO.setmode(GPIO.BCM)

# Set pins as output
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)

# Create PWM object
pwm = GPIO.PWM(PWMA, 10000)  # 1 kHz frequency
pwm.start(0)  # Start PWM with 0% duty cycle (motor off)

def motor_forward():
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)
    pwm.ChangeDutyCycle(100)  # Set motor speed (50%)

def motor_reverse():
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.HIGH)
    pwm.ChangeDutyCycle(100)  # Set motor speed (50%)

def motor_stop():
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.LOW)
    pwm.ChangeDutyCycle(0)  # Stop the motor

try:
    motor_forward()
    time.sleep(5)  # Run motor for 5 seconds
    motor_reverse()
    time.sleep(5)  # Run motor in reverse for 5 seconds
    motor_stop()
finally:
    GPIO.cleanup()  # Cleanup GPIO
