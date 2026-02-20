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

# Initialize with retry logic
def init_mpu6050(max_retries=3):
    for attempt in range(max_retries):
        try:
            bus = smbus2.SMBus(1)
            address = 0x68
            # Wake up MPU6050
            bus.write_byte_data(address, 0x6B, 0x00)
            time.sleep(0.1)
            # Configure accelerometer (±2g range)
            bus.write_byte_data(address, 0x1C, 0x00)
            # Configure gyroscope (±250°/s range)
            bus.write_byte_data(address, 0x1B, 0x00)
            time.sleep(0.1)
            print("✓ MPU6050 initialized successfully!")
            return bus, address
        except BlockingIOError:
            print(f"⚠ Attempt {attempt + 1}/{max_retries}: I2C bus busy, retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"⚠ Attempt {attempt + 1}/{max_retries}: Error - {e}")
            time.sleep(1)
    
    raise Exception("Failed to initialize MPU6050 after multiple attempts")


# Tremor ratio thresholds (frequency-based)
RATIO_SIGNIFICANT = 0.35  #  0.3-0.5
RATIO_MILD = 0.20         #  0.15-0.3

# RMS amplitude thresholds (acceleration-based)
RMS_ACCEL_SIGNIFICANT = 0.50  # m/s² -  0.4-0.6
RMS_ACCEL_MILD = 0.35         # m/s² -  0.3-0.4

# RMS amplitude thresholds (gyroscope-based)  
RMS_GYRO_SIGNIFICANT = 18     # °/s -  15-25
RMS_GYRO_MILD = 10            # °/s -  8-15


# Configuration - SPIRAL TEST SPECIFIC
SAMPLE_RATE = 100 
WINDOW_SECONDS = 10  
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

print("=" * 50)
print("MPU6050 Touchscreen Spiral Drawing Test")
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
filename = f"{patient_name.replace(' ', '_')}_spiral{test_number}_{timestamp}"

print(f"\nPatient: {patient_name}")
print(f"Test: Spiral Drawing #{test_number}")
print(f"Results will be saved to: {plots_folder}")
print()

# Initialize MPU6050 with retry logic
bus, address = init_mpu6050()

# Data buffers for accelerometer
buffer_ax = deque(maxlen=WINDOW_SIZE)
buffer_ay = deque(maxlen=WINDOW_SIZE)
buffer_az = deque(maxlen=WINDOW_SIZE)

# Data buffers for gyroscope
buffer_gx = deque(maxlen=WINDOW_SIZE)
buffer_gy = deque(maxlen=WINDOW_SIZE)
buffer_gz = deque(maxlen=WINDOW_SIZE)

print("=" * 50)
print("INSTRUCTIONS:")
print("Start drawing the spiral on touchscreen when data collection begins")
print("Draw slowly and steadily for the full 10 seconds")
print("=" * 50)
input("\nPress ENTER when ready to start...")
print()

print("Collecting data... DRAW THE SPIRAL NOW!")
print(f"Keep drawing for {WINDOW_SECONDS} seconds")
print()

# Helper function to read signed 16-bit value with retry logic
def read_word_2c_safe(addr, reg, max_retries=3):
    """Read word with retry logic for flaky connections"""
    for attempt in range(max_retries):
        try:
            high = bus.read_byte_data(addr, reg)
            low = bus.read_byte_data(addr, reg + 1)
            val = (high << 8) + low
            if val >= 0x8000:
                return -((65535 - val) + 1)
            else:
                return val
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(0.01)  # Short delay before retry
                continue
            else:
                raise  # Re-raise if all retries failed

# Collect data
start_time = time.time()
sample_count = 0

try:
    while len(buffer_ax) < WINDOW_SIZE:
        # Read accelerometer data (registers 0x3B to 0x40)
        accel_x = read_word_2c_safe(address, 0x3B)
        accel_y = read_word_2c_safe(address, 0x3D)
        accel_z = read_word_2c_safe(address, 0x3F)
        
        # Read gyroscope data (registers 0x43 to 0x48)
        gyro_x = read_word_2c_safe(address, 0x43)
        gyro_y = read_word_2c_safe(address, 0x45)
        gyro_z = read_word_2c_safe(address, 0x47)
        
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
        if sample_count % 100 == 0:
            elapsed = sample_count / SAMPLE_RATE
            remaining = WINDOW_SECONDS - elapsed
            print(f"Drawing... {remaining:.1f} seconds remaining")
        
        time.sleep(1.0 / SAMPLE_RATE)
    
    print(f"\nData collection complete! Analyzing...")
    
    # Convert to numpy arrays
    ax_data = np.array(buffer_ax)
    ay_data = np.array(buffer_ay)
    az_data = np.array(buffer_az)
    gx_data = np.array(buffer_gx)
    gy_data = np.array(buffer_gy)
    gz_data = np.array(buffer_gz)
    
    # Save raw data for later analysis
    raw_data_path = os.path.join(plots_folder, f"{filename}_raw.npz")
    np.savez(raw_data_path, 
         ax=ax_data, ay=ay_data, az=az_data,
         gx=gx_data, gy=gy_data, gz=gz_data,
         sample_rate=SAMPLE_RATE,
         patient_name=patient_name,
         test_number=test_number,
         timestamp=timestamp,
         test_type='spiral')
    print(f"Raw data saved: {raw_data_path}")
    
    # Calculate linear acceleration magnitude
    accel_magnitude = np.sqrt(ax_data**2 + ay_data**2 + az_data**2)
    
    # Calculate rotational velocity magnitude
    gyro_magnitude = np.sqrt(gx_data**2 + gy_data**2 + gz_data**2)
    
    # High-pass filter for ACTION TREMOR (2 Hz for spiral test)
    sos_accel = signal.butter(4, 2, 'hp', fs=SAMPLE_RATE, output='sos')
    filtered_accel = signal.sosfilt(sos_accel, accel_magnitude)
    
    sos_gyro = signal.butter(4, 2, 'hp', fs=SAMPLE_RATE, output='sos')
    filtered_gyro = signal.sosfilt(sos_gyro, gyro_magnitude)
    
    # Compute Power Spectral Density for accelerometer
    freqs_accel, psd_accel = signal.welch(filtered_accel, SAMPLE_RATE, nperseg=SAMPLE_RATE*2)
    
    # Compute Power Spectral Density for gyroscope
    freqs_gyro, psd_gyro = signal.welch(filtered_gyro, SAMPLE_RATE, nperseg=SAMPLE_RATE*2)
    
    # Find tremor in 4-6 Hz band (accelerometer) - ACTION TREMOR RANGE
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
    
    # TOUCHSCREEN-SPECIFIC METRICS
    # Use RMS amplitude (root mean square) for better measure
    tremor_rms_accel = np.sqrt(np.mean(filtered_accel**2))
    tremor_rms_gyro = np.sqrt(np.mean(filtered_gyro**2))
    
    # Peak-to-peak tremor variation
    tremor_p2p_accel = np.max(filtered_accel) - np.min(filtered_accel)
    tremor_p2p_gyro = np.max(filtered_gyro) - np.min(filtered_gyro)
    
    # Movement smoothness (lower jerk = smoother)
    jerk = np.abs(np.diff(accel_magnitude)) * SAMPLE_RATE
    smoothness_score = np.mean(jerk)
    
    print("\n" + "=" * 50)
    print("TOUCHSCREEN SPIRAL DRAWING TREMOR ANALYSIS")
    print("=" * 50)
    print(f"\nPatient: {patient_name}")
    print(f"Test: Spiral #{test_number}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nACTION TREMOR - LINEAR (Accelerometer):")
    print(f"  Peak frequency: {peak_freq_accel:.2f} Hz")
    print(f"  Tremor ratio: {tremor_ratio_accel:.3f}")
    print(f"  RMS amplitude: {tremor_rms_accel:.3f} m/s²")
    print(f"  Peak-to-peak: {tremor_p2p_accel:.3f} m/s²")
    
    print("\nACTION TREMOR - ROTATIONAL (Gyroscope):")
    print(f"  Peak frequency: {peak_freq_gyro:.2f} Hz")
    print(f"  Tremor ratio: {tremor_ratio_gyro:.3f}")
    print(f"  RMS amplitude: {tremor_rms_gyro:.3f} °/s")
    print(f"  Peak-to-peak: {tremor_p2p_gyro:.3f} °/s")
    
    print("\nDRAWING QUALITY:")
    print(f"  Movement smoothness: {smoothness_score:.3f} (lower = smoother)")
    
    # TOUCHSCREEN SPIRAL TEST ASSESSMENT with ADJUSTABLE THRESHOLDS
    print("\n" + "=" * 50)
    print("OVERALL ASSESSMENT (Touchscreen Drawing)")
    print("=" * 50)
    print(f"\nConfigured Thresholds:")
    print(f"  Significant tremor:")
    print(f"    - Ratio > {RATIO_SIGNIFICANT}")
    print(f"    - RMS Accel > {RMS_ACCEL_SIGNIFICANT} m/s²")
    print(f"    - RMS Gyro > {RMS_GYRO_SIGNIFICANT} °/s")
    print(f"  Mild tremor:")
    print(f"    - Ratio > {RATIO_MILD}")
    print(f"    - RMS Accel > {RMS_ACCEL_MILD} m/s²")
    print(f"    - RMS Gyro > {RMS_GYRO_MILD} °/s")
    
    tremor_status = ""
    
    # Multi-criteria detection (OR logic for sensitivity)
    high_tremor_ratio = (tremor_ratio_accel > RATIO_SIGNIFICANT or tremor_ratio_gyro > RATIO_SIGNIFICANT)
    correct_frequency = (4 <= peak_freq_accel <= 6 or 4 <= peak_freq_gyro <= 6)
    high_amplitude = (tremor_rms_accel > RMS_ACCEL_SIGNIFICANT or tremor_rms_gyro > RMS_GYRO_SIGNIFICANT)
    
    moderate_tremor_ratio = (tremor_ratio_accel > RATIO_MILD or tremor_ratio_gyro > RATIO_MILD)
    moderate_amplitude = (tremor_rms_accel > RMS_ACCEL_MILD or tremor_rms_gyro > RMS_GYRO_MILD)
    
    # Detailed diagnostic output
    print(f"\n📊 DIAGNOSTIC VALUES:")
    print(f"  Tremor ratio (accel): {tremor_ratio_accel:.3f} {'✓ HIGH' if tremor_ratio_accel > RATIO_SIGNIFICANT else '✓ MODERATE' if tremor_ratio_accel > RATIO_MILD else '  normal'}")
    print(f"  Tremor ratio (gyro): {tremor_ratio_gyro:.3f} {'✓ HIGH' if tremor_ratio_gyro > RATIO_SIGNIFICANT else '✓ MODERATE' if tremor_ratio_gyro > RATIO_MILD else '  normal'}")
    print(f"  RMS amplitude (accel): {tremor_rms_accel:.3f} m/s² {'✓ HIGH' if tremor_rms_accel > RMS_ACCEL_SIGNIFICANT else '✓ MODERATE' if tremor_rms_accel > RMS_ACCEL_MILD else '  normal'}")
    print(f"  RMS amplitude (gyro): {tremor_rms_gyro:.3f} °/s {'✓ HIGH' if tremor_rms_gyro > RMS_GYRO_SIGNIFICANT else '✓ MODERATE' if tremor_rms_gyro > RMS_GYRO_MILD else '  normal'}")
    print(f"  Peak frequency (accel): {peak_freq_accel:.2f} Hz {'✓ IN TREMOR BAND' if 4 <= peak_freq_accel <= 6 else '  outside band'}")
    print(f"  Peak frequency (gyro): {peak_freq_gyro:.2f} Hz {'✓ IN TREMOR BAND' if 4 <= peak_freq_gyro <= 6 else '  outside band'}")
    
    print(f"\n🔍 CRITERIA CHECK:")
    print(f"  High tremor ratio: {'✓ YES' if high_tremor_ratio else '✗ NO'}")
    print(f"  Correct frequency (4-6 Hz): {'✓ YES' if correct_frequency else '✗ NO'}")
    print(f"  High amplitude: {'✓ YES' if high_amplitude else '✗ NO'}")
    
    # Detection logic: Must meet frequency AND (ratio OR amplitude)
    if correct_frequency and (high_tremor_ratio or high_amplitude):
        tremor_status = "SIGNIFICANT ACTION TREMOR"
        print("\n  ⚠️  SIGNIFICANT ACTION TREMOR DETECTED")
        print("     High tremor activity in 4-6 Hz range")
        if high_tremor_ratio:
            print(f"     - Frequency component: {max(tremor_ratio_accel, tremor_ratio_gyro):.3f} (threshold: {RATIO_SIGNIFICANT})")
        if high_amplitude:
            print(f"     - Amplitude: {tremor_rms_accel:.3f} m/s² or {tremor_rms_gyro:.3f} °/s")
    elif correct_frequency and (moderate_tremor_ratio or moderate_amplitude):
        tremor_status = "MILD ACTION TREMOR"
        print("\n  ⚡ MILD ACTION TREMOR")
        print("     Slight tremor during drawing")
        if moderate_tremor_ratio:
            print(f"     - Frequency component: {max(tremor_ratio_accel, tremor_ratio_gyro):.3f} (threshold: {RATIO_MILD})")
        if moderate_amplitude:
            print(f"     - Amplitude: {tremor_rms_accel:.3f} m/s² or {tremor_rms_gyro:.3f} °/s")
    else:
        tremor_status = "NO SIGNIFICANT TREMOR"
        print("\n  ✓  NO SIGNIFICANT TREMOR")
        print("     Normal motion for touchscreen drawing")
        
        # Helpful feedback for tuning thresholds
        if not correct_frequency:
            print(f"     Note: Peak frequency outside tremor band (accel: {peak_freq_accel:.2f} Hz, gyro: {peak_freq_gyro:.2f} Hz)")
        elif tremor_ratio_accel > 0.1 or tremor_ratio_gyro > 0.1:
            print(f"     Note: Minor 4-6 Hz component present (ratio: {max(tremor_ratio_accel, tremor_ratio_gyro):.3f})")
        if tremor_rms_accel > 0.2 or tremor_rms_gyro > 5:
            print(f"     Note: Some movement detected but below tremor threshold")
    
    if smoothness_score > 2.0:
        print("  ⚠️  Reduced movement smoothness detected")
    
    # Threshold tuning suggestions
    print("\n💡 THRESHOLD TUNING SUGGESTIONS:")
    if tremor_status == "NO SIGNIFICANT TREMOR" and (tremor_ratio_accel > RATIO_MILD*0.7 or tremor_rms_accel > RMS_ACCEL_MILD*0.7):
        print("  If this should be detected as tremor, consider lowering thresholds by 20-30%")
    elif tremor_status != "NO SIGNIFICANT TREMOR":
        print("  If this is a false positive, consider raising thresholds by 20-30%")
    else:
        print("  Thresholds appear well-calibrated for this case")
    
    # Create plots
    time_axis = np.arange(len(accel_magnitude)) / SAMPLE_RATE
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    
    # Add patient info
    fig.suptitle(f'Touchscreen Spiral Test - {patient_name} - Test {test_number}\n{tremor_status}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Raw accelerometer magnitude
    axes[0].plot(time_axis, accel_magnitude, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Magnitude (m/s²)')
    axes[0].set_title('Raw Acceleration During Spiral Drawing')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Filtered accelerometer magnitude
    axes[1].plot(time_axis, filtered_accel, 'r-', linewidth=0.8)
    axes[1].set_ylabel('Filtered Magnitude (m/s²)')
    axes[1].set_title(f'Action Tremor (High-pass >2 Hz) - RMS: {tremor_rms_accel:.3f} m/s²')
    axes[1].axhline(y=RMS_ACCEL_SIGNIFICANT, color='orange', linestyle='--', alpha=0.5, label=f'Significant: {RMS_ACCEL_SIGNIFICANT}')
    axes[1].axhline(y=RMS_ACCEL_MILD, color='yellow', linestyle='--', alpha=0.5, label=f'Mild: {RMS_ACCEL_MILD}')
    axes[1].axhline(y=-RMS_ACCEL_SIGNIFICANT, color='orange', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-RMS_ACCEL_MILD, color='yellow', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Raw gyroscope magnitude
    axes[2].plot(time_axis, gyro_magnitude, 'g-', linewidth=0.8)
    axes[2].set_ylabel('Magnitude (°/s)')
    axes[2].set_title('Raw Gyroscope During Spiral Drawing')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Filtered gyroscope magnitude
    axes[3].plot(time_axis, filtered_gyro, 'm-', linewidth=0.8)
    axes[3].set_ylabel('Filtered Magnitude (°/s)')
    axes[3].set_title(f'Action Tremor (High-pass >2 Hz) - RMS: {tremor_rms_gyro:.3f} °/s')
    axes[3].axhline(y=RMS_GYRO_SIGNIFICANT, color='orange', linestyle='--', alpha=0.5, label=f'Significant: {RMS_GYRO_SIGNIFICANT}')
    axes[3].axhline(y=RMS_GYRO_MILD, color='yellow', linestyle='--', alpha=0.5, label=f'Mild: {RMS_GYRO_MILD}')
    axes[3].axhline(y=-RMS_GYRO_SIGNIFICANT, color='orange', linestyle='--', alpha=0.5)
    axes[3].axhline(y=-RMS_GYRO_MILD, color='yellow', linestyle='--', alpha=0.5)
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Accelerometer PSD
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
    
    # Plot 6: Gyroscope PSD
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
    
    # Save text report
    report_path = os.path.join(plots_folder, f"{filename}.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TOUCHSCREEN SPIRAL DRAWING TREMOR ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Patient: {patient_name}\n")
        f.write(f"Test Number: Spiral #{test_number}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Rate: {SAMPLE_RATE} Hz\n")
        f.write(f"Window Duration: {WINDOW_SECONDS} seconds\n")
        f.write(f"Filter Type: High-pass 2 Hz (Action Tremor)\n\n")
        
        f.write("CONFIGURED THRESHOLDS:\n")
        f.write(f"  Significant tremor:\n")
        f.write(f"    - Ratio > {RATIO_SIGNIFICANT}\n")
        f.write(f"    - RMS Accel > {RMS_ACCEL_SIGNIFICANT} m/s²\n")
        f.write(f"    - RMS Gyro > {RMS_GYRO_SIGNIFICANT} °/s\n")
        f.write(f"  Mild tremor:\n")
        f.write(f"    - Ratio > {RATIO_MILD}\n")
        f.write(f"    - RMS Accel > {RMS_ACCEL_MILD} m/s²\n")
        f.write(f"    - RMS Gyro > {RMS_GYRO_MILD} °/s\n\n")
        
        f.write("ACTION TREMOR - LINEAR (Accelerometer):\n")
        f.write(f"  Peak frequency: {peak_freq_accel:.2f} Hz\n")
        f.write(f"  Tremor ratio: {tremor_ratio_accel:.3f}\n")
        f.write(f"  RMS amplitude: {tremor_rms_accel:.3f} m/s²\n")
        f.write(f"  Peak-to-peak: {tremor_p2p_accel:.3f} m/s²\n\n")
        
        f.write("ACTION TREMOR - ROTATIONAL (Gyroscope):\n")
        f.write(f"  Peak frequency: {peak_freq_gyro:.2f} Hz\n")
        f.write(f"  Tremor ratio: {tremor_ratio_gyro:.3f}\n")
        f.write(f"  RMS amplitude: {tremor_rms_gyro:.3f} °/s\n")
        f.write(f"  Peak-to-peak: {tremor_p2p_gyro:.3f} °/s\n\n")
        
        f.write("DRAWING QUALITY:\n")
        f.write(f"  Movement smoothness: {smoothness_score:.3f}\n\n")
        
        f.write("OVERALL ASSESSMENT:\n")
        f.write(f"  Status: {tremor_status}\n")
    
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
