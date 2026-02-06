import smbus2
import time
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from collections import deque
import os
from datetime import datetime

bus = smbus2.SMBus(1)
address = 0x68 

# Configuration
SAMPLE_RATE = 100 
WINDOW_SECONDS = 4  
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

print("=" * 50)
print("MPU6050 Tremor Detection System")
print("=" * 50)

# Get user input
patient_name = input("\nEnter patient name: ").strip()
test_number = input("Enter test number: ").strip()

if not patient_name:
    patient_name = "Unknown"
if not test_number:
    test_number = "1"

# Use the plots folder directly
plots_folder = "/home/pi/Desktop/tremor_detection/plots"

# Generate timestamp and filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{patient_name.replace(' ', '_')}_test{test_number}_{timestamp}"

print(f"\nPatient: {patient_name}")
print(f"Test: {test_number}")
print(f"Results will be saved to: {plots_folder}")
print()

# Wake up MPU6050
bus.write_byte_data(address, 0x6B, 0x00)
time.sleep(0.1)

# Configure accelerometer (±2g range)
bus.write_byte_data(address, 0x1C, 0x00)
# Configure gyroscope (±250°/s range)
bus.write_byte_data(address, 0x1B, 0x00)
time.sleep(0.1)

# Data buffers for accelerometer
buffer_ax = deque(maxlen=WINDOW_SIZE)
buffer_ay = deque(maxlen=WINDOW_SIZE)
buffer_az = deque(maxlen=WINDOW_SIZE)

# Data buffers for gyroscope
buffer_gx = deque(maxlen=WINDOW_SIZE)
buffer_gy = deque(maxlen=WINDOW_SIZE)
buffer_gz = deque(maxlen=WINDOW_SIZE)

print("Collecting data... Hold sensor steady or simulate tremor")
print(f"Will analyze after {WINDOW_SECONDS} seconds")
print()

# Helper function to read signed 16-bit value
def read_word_2c(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    val = (high << 8) + low
    if val >= 0x8000:
        return -((65535 - val) + 1)
    else:
        return val

# Collect data
start_time = time.time()
sample_count = 0

try:
    while len(buffer_ax) < WINDOW_SIZE:
        # Read accelerometer data (registers 0x3B to 0x40)
        accel_x = read_word_2c(address, 0x3B)
        accel_y = read_word_2c(address, 0x3D)
        accel_z = read_word_2c(address, 0x3F)
        
        # Read gyroscope data (registers 0x43 to 0x48)
        gyro_x = read_word_2c(address, 0x43)
        gyro_y = read_word_2c(address, 0x45)
        gyro_z = read_word_2c(address, 0x47)
        
        # Convert to physical units
        # Accelerometer: ±2g range, sensitivity = 16384 LSB/g
        ax = (accel_x / 16384.0) * 9.81  
        ay = (accel_y / 16384.0) * 9.81
        az = (accel_z / 16384.0) * 9.81
        
        # Gyroscope: ±250°/s range, sensitivity = 131 LSB/(°/s)
        gx = gyro_x / 131.0  
        gy = gyro_y / 131.0
        gz = gyro_z / 131.0
        
        buffer_ax.append(ax)
        buffer_ay.append(ay)
        buffer_az.append(az)
        buffer_gx.append(gx)
        buffer_gy.append(gy)
        buffer_gz.append(gz)
        
        sample_count += 1
        if sample_count % 50 == 0:
            print(f"Collected {sample_count}/{WINDOW_SIZE} samples...")
        
        time.sleep(1.0 / SAMPLE_RATE)
    
    print(f"\nData collection complete! Analyzing...")
    
    # Convert to numpy arrays
    ax_data = np.array(buffer_ax)
    ay_data = np.array(buffer_ay)
    az_data = np.array(buffer_az)
    gx_data = np.array(buffer_gx)
    gy_data = np.array(buffer_gy)
    gz_data = np.array(buffer_gz)
    
    # Calculate linear acceleration magnitude
    accel_magnitude = np.sqrt(ax_data**2 + ay_data**2 + az_data**2)
    
    # Calculate rotational velocity magnitude
    gyro_magnitude = np.sqrt(gx_data**2 + gy_data**2 + gz_data**2)
    
    # High-pass filter to remove gravity and slow movements
    sos_accel = signal.butter(4, 1, 'hp', fs=SAMPLE_RATE, output='sos')
    filtered_accel = signal.sosfilt(sos_accel, accel_magnitude)
    
    sos_gyro = signal.butter(4, 1, 'hp', fs=SAMPLE_RATE, output='sos')
    filtered_gyro = signal.sosfilt(sos_gyro, gyro_magnitude)
    
    # Compute Power Spectral Density for accelerometer
    freqs_accel, psd_accel = signal.welch(filtered_accel, SAMPLE_RATE, nperseg=SAMPLE_RATE*2)
    
    # Compute Power Spectral Density for gyroscope
    freqs_gyro, psd_gyro = signal.welch(filtered_gyro, SAMPLE_RATE, nperseg=SAMPLE_RATE*2)
    
    # Find tremor in 4-6 Hz band (accelerometer)
    tremor_band = (freqs_accel >= 4) & (freqs_accel <= 6)
    tremor_power_accel = np.sum(psd_accel[tremor_band])
    total_power_accel = np.sum(psd_accel[freqs_accel <= 15])
    tremor_ratio_accel = tremor_power_accel / total_power_accel if total_power_accel > 0 else 0
    
    # Find tremor in 4-6 Hz band (gyroscope)
    tremor_band_gyro = (freqs_gyro >= 4) & (freqs_gyro <= 6)
    tremor_power_gyro = np.sum(psd_gyro[tremor_band_gyro])
    total_power_gyro = np.sum(psd_gyro[freqs_gyro <= 15])
    tremor_ratio_gyro = tremor_power_gyro / total_power_gyro if total_power_gyro > 0 else 0
    
    # Find peak frequencies
    peak_idx_accel = np.argmax(psd_accel[freqs_accel <= 10])
    peak_freq_accel = freqs_accel[freqs_accel <= 10][peak_idx_accel]
    
    peak_idx_gyro = np.argmax(psd_gyro[freqs_gyro <= 10])
    peak_freq_gyro = freqs_gyro[freqs_gyro <= 10][peak_idx_gyro]
    
    print("\n" + "=" * 50)
    print("TREMOR ANALYSIS RESULTS")
    print("=" * 50)
    print(f"\nPatient: {patient_name}")
    print(f"Test: {test_number}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nLINEAR TREMOR (Accelerometer):")
    print(f"  Peak frequency: {peak_freq_accel:.2f} Hz")
    print(f"  Tremor power ratio: {tremor_ratio_accel:.3f}")
    print(f"  Tremor band power: {tremor_power_accel:.2e}")
    
    print("\nROTATIONAL TREMOR (Gyroscope):")
    print(f"  Peak frequency: {peak_freq_gyro:.2f} Hz")
    print(f"  Tremor power ratio: {tremor_ratio_gyro:.3f}")
    print(f"  Tremor band power: {tremor_power_gyro:.2e}")
    
    # Combined tremor detection
    combined_ratio = (tremor_ratio_accel + tremor_ratio_gyro) / 2
    
    print("\nOVERALL ASSESSMENT:")
    tremor_status = ""
    if (tremor_ratio_accel > 0.3 or tremor_ratio_gyro > 0.3) and \
       (4 <= peak_freq_accel <= 6 or 4 <= peak_freq_gyro <= 6):
        tremor_status = "TREMOR DETECTED"
        print("️  TREMOR DETECTED - Significant 4-6 Hz activity")
        if tremor_ratio_gyro > tremor_ratio_accel:
            print("   Predominantly ROTATIONAL tremor (pill-rolling pattern)")
        else:
            print("   Predominantly LINEAR tremor")
    elif combined_ratio > 0.15:
        tremor_status = "POSSIBLE TREMOR"
        print(" POSSIBLE TREMOR - Elevated 4-6 Hz activity")
    else:
        tremor_status = "NO TREMOR"
        print("  NO TREMOR - Normal movement pattern")
    
    # Create plots NEED SIX!!!!!!!!!!!!!!
    time_axis = np.arange(len(accel_magnitude)) / SAMPLE_RATE
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    
    # Add patient info
    fig.suptitle(f'Tremor Analysis - {patient_name} - Test {test_number}\n{tremor_status}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1 Raw accelerometer magnitude
    axes[0].plot(time_axis, accel_magnitude, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Magnitude (m/s²)')
    axes[0].set_title('Raw Acceleration Magnitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2 Filtered accelerometer magnitude
    axes[1].plot(time_axis, filtered_accel, 'r-', linewidth=0.8)
    axes[1].set_ylabel('Filtered Magnitude (m/s²)')
    axes[1].set_title('Filtered Acceleration (High-pass >1 Hz)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3 Raw gyroscope magnitude
    axes[2].plot(time_axis, gyro_magnitude, 'g-', linewidth=0.8)
    axes[2].set_ylabel('Magnitude (°/s)')
    axes[2].set_title('Raw Gyroscope Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4 Filtered gyroscope magnitude
    axes[3].plot(time_axis, filtered_gyro, 'm-', linewidth=0.8)
    axes[3].set_ylabel('Filtered Magnitude (°/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Filtered Gyroscope (High-pass >1 Hz)')
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5 Accelerometer PSD
    axes[4].semilogy(freqs_accel, psd_accel, 'b-', linewidth=1.5)
    axes[4].axvspan(4, 6, alpha=0.3, color='red', label='Tremor band (4-6 Hz)')
    axes[4].axvline(peak_freq_accel, color='blue', linestyle='--', 
                    label=f'Peak: {peak_freq_accel:.2f} Hz')
    axes[4].set_xlabel('Frequency (Hz)')
    axes[4].set_ylabel('Power Spectral Density')
    axes[4].set_xlim(0, 15)
    axes[4].set_title(f'Accelerometer Frequency Analysis (Tremor Ratio: {tremor_ratio_accel:.3f})')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # Plot 6 Gyroscope PSD
    axes[5].semilogy(freqs_gyro, psd_gyro, 'g-', linewidth=1.5)
    axes[5].axvspan(4, 6, alpha=0.3, color='red', label='Tremor band (4-6 Hz)')
    axes[5].axvline(peak_freq_gyro, color='green', linestyle='--', 
                    label=f'Peak: {peak_freq_gyro:.2f} Hz')
    axes[5].set_xlabel('Frequency (Hz)')
    axes[5].set_ylabel('Power Spectral Density')
    axes[5].set_xlim(0, 15)
    axes[5].set_title(f'Gyroscope Frequency Analysis (Tremor Ratio: {tremor_ratio_gyro:.3f})')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot in plots folder
    plot_path = os.path.join(plots_folder, f"{filename}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    
    report_path = os.path.join(plots_folder, f"{filename}.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TREMOR ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Patient: {patient_name}\n")
        f.write(f"Test Number: {test_number}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Rate: {SAMPLE_RATE} Hz\n")
        f.write(f"Window Duration: {WINDOW_SECONDS} seconds\n\n")
        f.write("LINEAR TREMOR (Accelerometer):\n")
        f.write(f"  Peak frequency: {peak_freq_accel:.2f} Hz\n")
        f.write(f"  Tremor power ratio: {tremor_ratio_accel:.3f}\n")
        f.write(f"  Tremor band power: {tremor_power_accel:.2e}\n\n")
        f.write("ROTATIONAL TREMOR (Gyroscope):\n")
        f.write(f"  Peak frequency: {peak_freq_gyro:.2f} Hz\n")
        f.write(f"  Tremor power ratio: {tremor_ratio_gyro:.3f}\n")
        f.write(f"  Tremor band power: {tremor_power_gyro:.2e}\n\n")
        f.write("OVERALL ASSESSMENT:\n")
        f.write(f"  Status: {tremor_status}\n")
        if (tremor_ratio_accel > 0.3 or tremor_ratio_gyro > 0.3) and \
           (4 <= peak_freq_accel <= 6 or 4 <= peak_freq_gyro <= 6):
            if tremor_ratio_gyro > tremor_ratio_accel:
                f.write("  Type: Predominantly ROTATIONAL tremor (pill-rolling pattern)\n")
            else:
                f.write("  Type: Predominantly LINEAR tremor\n")
    
    print("\n" + "=" * 50)
    print("RESULTS SAVED")
    print("=" * 50)
    print(f"Plot: {plot_path}")
    print(f"Report: {report_path}")
    print("\nDone!")
    
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
