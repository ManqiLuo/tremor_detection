import board
import busio
import adafruit_mma8451
import time

# Initialize I2C and sensor
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_mma8451.MMA8451(i2c, address=0x1C)
sensor.range = adafruit_mma8451.RANGE_2G  # ±2g for tremor detection

print("MMA845X Accelerometer Test")
print("Reading acceleration values...")
print("Press Ctrl+C to stop\n")

try:
    for i in range(20):
        x, y, z = sensor.acceleration
        print(f"X: {x:6.2f}  Y: {y:6.2f}  Z: {z:6.2f} m/s²")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nStopped!")
