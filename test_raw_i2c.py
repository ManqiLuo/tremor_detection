import smbus2
import time

bus = smbus2.SMBus(1)
address = 0x1C

print("Testing raw I2C communication with device at 0x1C...")

try:
    # Try to read WHO_AM_I register (0x0D)
    who_am_i = bus.read_byte_data(address, 0x0D)
    print(f"WHO_AM_I register: 0x{who_am_i:02X}")
    
    # MMA8451 should return 0x1A
    # MMA8452 should return 0x2A
    # MMA8453 should return 0x3A
    
    if who_am_i == 0x1A:
        print("This is an MMA8451!")
    elif who_am_i == 0x2A:
        print("This is an MMA8452!")
    elif who_am_i == 0x3A:
        print("This is an MMA8453!")
    else:
        print(f"Unknown device ID: 0x{who_am_i:02X}")
        print("This might not be an MMA845X sensor")
        
except Exception as e:
    print(f"Error: {e}")
