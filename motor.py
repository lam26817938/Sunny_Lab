import RPi.GPIO as GPIO
import time

# Set up GPIO mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins for PWM and sensors
ENA_PIN = 12  # PWM control for motor
IN1_PIN = 23  # Motor direction control pin 1
IN2_PIN = 24  # Motor direction control pin 2
SENSOR_1_PIN = 17  # Sensor 1 GPIO pin
SENSOR_2_PIN = 27  # Sensor 2 GPIO pin

# Set up motor control pins
GPIO.setup(ENA_PIN, GPIO.OUT)
GPIO.setup(IN1_PIN, GPIO.OUT)
GPIO.setup(IN2_PIN, GPIO.OUT)
GPIO.setup(SENSOR_1_PIN, GPIO.IN)
GPIO.setup(SENSOR_2_PIN, GPIO.IN)

# Set up PWM on ENA pin
pwm = GPIO.PWM(ENA_PIN, 100)  # 100 Hz frequency for PWM
pwm.start(0)  # Start PWM with 0% duty cycle (motor off)

# Target speed in arbitrary units (you can adjust this based on your requirements)
target_speed = 50

# Helper function to set motor speed and direction
def set_motor_speed(speed, direction="forward"):
    pwm.ChangeDutyCycle(speed)  # Adjust PWM duty cycle for speed
    if direction == "forward":
        GPIO.output(IN1_PIN, GPIO.HIGH)
        GPIO.output(IN2_PIN, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(IN1_PIN, GPIO.LOW)
        GPIO.output(IN2_PIN, GPIO.HIGH)

# Function to calculate speed based on sensor input
def calculate_speed():
    # Wait for Sensor 1 to change state
    while GPIO.input(SENSOR_1_PIN) == 0:
        pass
    start_time = time.time()
    
    # Wait for Sensor 2 to change state
    while GPIO.input(SENSOR_2_PIN) == 0:
        pass
    end_time = time.time()
    
    # Calculate time difference between sensor activations
    time_diff = end_time - start_time
    if time_diff == 0:
        return 0
    speed = 1 / time_diff  # Inverse of time difference as speed approximation
    
    return speed

# Function to adjust motor speed to reach target speed
def adjust_speed():
    try:
        while True:
            actual_speed = calculate_speed()
            print(f"Actual Speed: {actual_speed:.2f}, Target Speed: {target_speed}")
            
            # Compare actual speed with target speed
            if actual_speed < target_speed:
                # Increase PWM duty cycle to speed up
                current_duty_cycle = pwm.duty_cycle if hasattr(pwm, 'duty_cycle') else 50
                new_duty_cycle = min(100, current_duty_cycle + 5)
                pwm.ChangeDutyCycle(new_duty_cycle)
                print(f"Increasing PWM to {new_duty_cycle}%")
            elif actual_speed > target_speed:
                # Decrease PWM duty cycle to slow down
                current_duty_cycle = pwm.duty_cycle if hasattr(pwm, 'duty_cycle') else 50
                new_duty_cycle = max(0, current_duty_cycle - 5)
                pwm.ChangeDutyCycle(new_duty_cycle)
                print(f"Decreasing PWM to {new_duty_cycle}%")
            else:
                print("Maintaining current PWM")
            
            # Small delay to avoid too frequent adjustments
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        pwm.stop()
        GPIO.cleanup()

# Main function to run motor and adjust speed
def main():
    set_motor_speed(50, "forward")  # Start motor at 50% speed
 #   adjust_speed()  # Continuously adjust to maintain target speed

# Run the main function
if __name__ == "__main__":
    main()
