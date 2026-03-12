import os
import csv
import re
import numpy as np
from scipy import signal
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
EXPORT_DIR  = os.path.join(os.path.expanduser("~"), "Desktop", "export")
AGE_GROUPS  = ["18-24", "41-50", "51-60", "unknow"]

# Tremor thresholds
RATIO_SIGNIFICANT     = 0.35
RATIO_MILD            = 0.20
RMS_ACCEL_SIGNIFICANT = 0.50
RMS_ACCEL_MILD        = 0.35
RMS_GYRO_SIGNIFICANT  = 18.0
RMS_GYRO_MILD         = 10.0


def parse_ts(s):
    if not s or str(s).strip() == '':
        return None
    s = str(s).strip()
    s = re.sub(r'\+00:00$', '+00:00', s)
    s = re.sub(r'\+00$',    '+00:00', s)
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def safe_float(val):
    try:
        v = str(val).strip()
        if v == '' or v.lower() == 'none':
            return None
        return float(v)
    except:
        return None

def safe_bool(val):
    return str(val).strip().lower() in ('true', '1', 't', 'yes')

def remove_artifacts(data, threshold_multiplier=3.0):
    """
    Replace artifact samples (>threshold_multiplier * median) with
    linearly interpolated values so they don't corrupt the PSD.
    Returns cleaned array and artifact mask.
    """
    data      = np.array(data, dtype=float)
    threshold = np.median(data) * threshold_multiplier
    art_mask  = data > threshold

    if not np.any(art_mask):
        return data, art_mask

    clean = data.copy()
    idxs  = np.where(art_mask)[0]

    # Group into contiguous runs and interpolate each run
    runs = []
    run_start = idxs[0]
    for i in range(1, len(idxs)):
        if idxs[i] - idxs[i-1] > 1:
            runs.append((run_start, idxs[i-1]))
            run_start = idxs[i]
    runs.append((run_start, idxs[-1]))

    for (start, end) in runs:
        left  = clean[start - 1] if start > 0             else 0.0
        right = clean[end   + 1] if end < len(clean) - 1  else 0.0
        clean[start:end+1] = np.linspace(left, right, end - start + 1)

    return clean, art_mask

def analyze_tester(session_id, tester_path, out_base):
    print("\n" + "=" * 55)
    print(f"Session: {session_id}")
    print("=" * 55)
    # Read game_results.csv — look for spiral
    game_csv = os.path.join(tester_path, "game_results.csv")
    if not os.path.exists(game_csv):
        # Try case-insensitive search
        for f in os.listdir(tester_path):
            if f.lower() == "game_results.csv":
                game_csv = os.path.join(tester_path, f)
                break
        else:
            print(f"  SKIP: game_results.csv not found")
            print(f"  Files in folder: {os.listdir(tester_path)}")
            return "skipped"

    spiral_start = None
    spiral_end   = None

    with open(game_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            if row.get('game_name', '').lower() == 'spiral':
                spiral_start = parse_ts(row.get('started_at',   ''))
                spiral_end   = parse_ts(row.get('completed_at', ''))
                break

    if spiral_start is None or spiral_end is None:
        print(f"  SKIP: Spiral timestamps not found")
        return "skipped"

    duration = (spiral_end - spiral_start).total_seconds()
    print(f"  Spiral: {spiral_start.strftime('%H:%M:%S')} - "
          f"{spiral_end.strftime('%H:%M:%S')}  ({duration:.1f}s)")

    # Read sensor_timeseries.csv
    sensor_csv = os.path.join(tester_path, "sensor_timeseries.csv")
    if not os.path.exists(sensor_csv):
        for f in os.listdir(tester_path):
            if f.lower() == "sensor_timeseries.csv":
                sensor_csv = os.path.join(tester_path, f)
                break
        else:
            print(f"  SKIP: sensor_timeseries.csv not found")
            print(f"  Files in folder: {os.listdir(tester_path)}")
            return "skipped"

    ax_data = []; ay_data = []; az_data = []
    gx_data = []; gy_data = []; gz_data = []
    ts_list = []

    with open(sensor_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            ts = parse_ts(row.get('timestamp', ''))
            if ts is None or ts < spiral_start or ts > spiral_end:
                continue

            accel_valid = safe_bool(row.get('accel_is_valid', 'false'))
            ax = safe_float(row.get('accel_x'))
            ay = safe_float(row.get('accel_y'))
            az = safe_float(row.get('accel_z'))

            gyro_valid = safe_bool(row.get('gyro_is_valid', 'false'))
            gx = safe_float(row.get('gyro_x'))
            gy = safe_float(row.get('gyro_y'))
            gz = safe_float(row.get('gyro_z'))

            if accel_valid and ax is not None and ay is not None and az is not None:
                ax_data.append(ax); ay_data.append(ay); az_data.append(az)
                ts_list.append(ts)
                if gyro_valid and gx is not None and gy is not None and gz is not None:
                    gx_data.append(gx); gy_data.append(gy); gz_data.append(gz)
                else:
                    gx_data.append(0.0); gy_data.append(0.0); gz_data.append(0.0)

    if len(ax_data) == 0:
        print(f"  SKIP: No valid accel data in spiral window")
        return "skipped"

    if len(ts_list) >= 2:
        total_secs  = (ts_list[-1] - ts_list[0]).total_seconds()
        SAMPLE_RATE = int(round(len(ts_list) / total_secs)) if total_secs > 0 else 100
    else:
        SAMPLE_RATE = 100

    print(f"  Samples: {len(ax_data)}  |  Rate: {SAMPLE_RATE} Hz")

    ax_data = np.array(ax_data); ay_data = np.array(ay_data); az_data = np.array(az_data)
    gx_data = np.array(gx_data); gy_data = np.array(gy_data); gz_data = np.array(gz_data)

    # Raw magnitudes (for plotting only)
    accel_magnitude_raw = np.sqrt(ax_data**2 + ay_data**2 + az_data**2)
    gyro_magnitude_raw  = np.sqrt(gx_data**2 + gy_data**2 + gz_data**2)

    # Remove artifacts BEFORE filtering and PSD
    # Interpolate over spikes so they don't corrupt frequency analysis
    accel_magnitude_clean, accel_art_mask = remove_artifacts(accel_magnitude_raw, 3.0)
    gyro_magnitude_clean,  gyro_art_mask  = remove_artifacts(gyro_magnitude_raw,  3.0)

    n_accel_artifacts = int(np.sum(accel_art_mask))
    n_gyro_artifacts  = int(np.sum(gyro_art_mask))
    has_artifacts     = n_accel_artifacts > 0 or n_gyro_artifacts > 0

    # Cluster artifact events for plot annotation (from raw accel)
    artifact_sample_idx = np.where(accel_art_mask)[0]
    artifact_times_s    = artifact_sample_idx / SAMPLE_RATE
    artifact_events     = []
    if len(artifact_times_s) > 0:
        event_start    = artifact_times_s[0]
        event_peak_idx = artifact_sample_idx[0]
        for i in range(1, len(artifact_times_s)):
            if artifact_times_s[i] - artifact_times_s[i-1] > 0.5:
                artifact_events.append({'time': (event_start + artifact_times_s[i-1]) / 2,
                                        'peak_idx': event_peak_idx})
                event_start    = artifact_times_s[i]
                event_peak_idx = artifact_sample_idx[i]
            else:
                if accel_magnitude_raw[artifact_sample_idx[i]] > accel_magnitude_raw[event_peak_idx]:
                    event_peak_idx = artifact_sample_idx[i]
        artifact_events.append({'time': (event_start + artifact_times_s[-1]) / 2,
                                 'peak_idx': event_peak_idx})

    artifact_note = ("MOVEMENT ARTIFACT at: " +
                     ", ".join([f"~{e['time']:.1f}s" for e in artifact_events])
                     ) if has_artifacts else ""
    if has_artifacts:
        print(f"  Artifacts removed: {n_accel_artifacts} accel + {n_gyro_artifacts} gyro samples interpolated")

    # Tremor analysis on CLEANED data
    # Bandpass 1-15 Hz: removes gravity/drift (low) and noise (high)
    # Keeps only tremor-relevant range, reduces RMS inflation from broadband noise
    nyq = SAMPLE_RATE / 2
    bp_low  = max(1.0  / nyq, 0.001)
    bp_high = min(15.0 / nyq, 0.999)
    b_accel, a_accel = signal.butter(4, [bp_low, bp_high], btype='band')
    filtered_accel   = signal.filtfilt(b_accel, a_accel, accel_magnitude_clean)
    b_gyro,  a_gyro  = signal.butter(4, [bp_low, bp_high], btype='band')
    filtered_gyro    = signal.filtfilt(b_gyro,  a_gyro,  gyro_magnitude_clean)

    nperseg = min(SAMPLE_RATE * 2, len(filtered_accel))
    freqs_accel, psd_accel = signal.welch(filtered_accel, SAMPLE_RATE, nperseg=nperseg)
    freqs_gyro,  psd_gyro  = signal.welch(filtered_gyro,  SAMPLE_RATE, nperseg=nperseg)

    tremor_mask_a      = (freqs_accel >= 4) & (freqs_accel <= 6)
    tremor_power_accel = np.sum(psd_accel[tremor_mask_a])
    total_power_accel  = np.sum(psd_accel[freqs_accel <= 15])
    tremor_ratio_accel = tremor_power_accel / total_power_accel if total_power_accel > 0 else 0

    tremor_mask_g     = (freqs_gyro >= 4) & (freqs_gyro <= 6)
    tremor_power_gyro = np.sum(psd_gyro[tremor_mask_g])
    total_power_gyro  = np.sum(psd_gyro[freqs_gyro <= 15])
    tremor_ratio_gyro = tremor_power_gyro / total_power_gyro if total_power_gyro > 0 else 0

    mask10a         = freqs_accel <= 10
    peak_freq_accel = freqs_accel[mask10a][np.argmax(psd_accel[mask10a])]
    mask10g         = freqs_gyro <= 10
    peak_freq_gyro  = freqs_gyro[mask10g][np.argmax(psd_gyro[mask10g])]

    tremor_rms_accel = np.sqrt(np.mean(filtered_accel**2))
    tremor_rms_gyro  = np.sqrt(np.mean(filtered_gyro**2))
    tremor_p2p_accel = np.max(filtered_accel) - np.min(filtered_accel)
    tremor_p2p_gyro  = np.max(filtered_gyro)  - np.min(filtered_gyro)
    jerk             = np.abs(np.diff(accel_magnitude_clean)) * SAMPLE_RATE
    smoothness_score = np.mean(jerk)


    # Peak prominence check:
    # A real tremor peak should be significantly higher than
    # the surrounding non-tremor bands (not just a flat spectrum).
    # Compute mean PSD in neighbouring bands 1-4 Hz and 6-10 Hz,
    # then check if the 4-6 Hz band is at least 2x that baseline.

    def peak_is_prominent(freqs, psd, band_low=4, band_high=6, factor=2.0):
        tremor_mask  = (freqs >= band_low)  & (freqs <= band_high)
        below_mask   = (freqs >= 1)         & (freqs < band_low)
        above_mask   = (freqs > band_high)  & (freqs <= 10)
        tremor_mean  = np.mean(psd[tremor_mask])  if np.any(tremor_mask)  else 0
        baseline     = np.mean(np.concatenate([
                            psd[below_mask] if np.any(below_mask) else np.array([]),
                            psd[above_mask] if np.any(above_mask) else np.array([])
                        ])) if (np.any(below_mask) or np.any(above_mask)) else 0
        return (tremor_mean > factor * baseline) if baseline > 0 else False

    accel_peak_prominent = peak_is_prominent(freqs_accel, psd_accel)
    gyro_peak_prominent  = peak_is_prominent(freqs_gyro,  psd_gyro)

    # Assessment — now requires BOTH frequency AND prominent peak
    high_tremor_ratio     = tremor_ratio_accel > RATIO_SIGNIFICANT or tremor_ratio_gyro > RATIO_SIGNIFICANT
    correct_frequency     = 4 <= peak_freq_accel <= 6 or 4 <= peak_freq_gyro <= 6
    prominent_peak        = accel_peak_prominent or gyro_peak_prominent
    high_amplitude        = tremor_rms_accel > RMS_ACCEL_SIGNIFICANT or tremor_rms_gyro > RMS_GYRO_SIGNIFICANT
    moderate_tremor_ratio = tremor_ratio_accel > RATIO_MILD or tremor_ratio_gyro > RATIO_MILD
    moderate_amplitude    = tremor_rms_accel > RMS_ACCEL_MILD or tremor_rms_gyro > RMS_GYRO_MILD

    # Must have: correct frequency band AND prominent peak AND (high ratio OR high amplitude)
    if correct_frequency and prominent_peak and (high_tremor_ratio or high_amplitude):
        tremor_status = "SIGNIFICANT ACTION TREMOR"
    elif correct_frequency and prominent_peak and (moderate_tremor_ratio or moderate_amplitude):
        tremor_status = "MILD ACTION TREMOR"
    else:
        tremor_status = "NO SIGNIFICANT TREMOR"

    print(f"  Tremor ratio accel: {tremor_ratio_accel:.3f}  |  gyro: {tremor_ratio_gyro:.3f}")
    print(f"  RMS accel: {tremor_rms_accel:.3f} m/s²  |  gyro: {tremor_rms_gyro:.3f} deg/s")
    print(f"  Peak freq accel: {peak_freq_accel:.2f} Hz  |  gyro: {peak_freq_gyro:.2f} Hz")
    print(f"  Peak prominent accel: {accel_peak_prominent}  |  gyro: {gyro_peak_prominent}")
    if has_artifacts:
        print(f"  NOTE: {artifact_note}")
    print(f"  --> {tremor_status}")

    # Output folder
    out_folder = os.path.join(out_base, f"{session_id}_tremor_detection")
    os.makedirs(out_folder, exist_ok=True)
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"{session_id}_{run_ts}"

    # Plot — raw signals shown as-is, cleaned used for analysis
    time_axis = np.arange(len(accel_magnitude_raw)) / SAMPLE_RATE
    fig, axes = plt.subplots(6, 1, figsize=(14, 20))
    artifact_title_note = f"\n{artifact_note} (interpolated for analysis)" if has_artifacts else ""
    fig.suptitle(
        f"Spiral Tremor Analysis  |  Session: {session_id[:8]}...  |  Age: {AGE_GROUP}\n"
        f"{tremor_status}{artifact_title_note}\n"
        f"Window: {spiral_start.strftime('%H:%M:%S')} - {spiral_end.strftime('%H:%M:%S')}  "
        f"({duration:.1f}s)   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=11, fontweight='bold'
    )

    # Plot 1: Raw accel — show original with artifact markers
    raw_title = 'Raw Acceleration Magnitude  (artifacts marked in orange — interpolated for analysis)'
    axes[0].plot(time_axis, accel_magnitude_raw, 'b-', linewidth=0.6)
    for e in artifact_events:
        axes[0].axvline(e['time'], color='orange', linewidth=1.5, linestyle='--', alpha=0.8)
        axes[0].annotate(f"Artifact\n~{e['time']:.1f}s",
                         xy=(e['time'], accel_magnitude_raw[e['peak_idx']]),
                         xytext=(e['time'] + 0.5, accel_magnitude_raw[e['peak_idx']] * 1.01),
                         fontsize=7, color='orange',
                         arrowprops=dict(arrowstyle='->', color='orange', lw=1.0))
    axes[0].set_ylabel('Magnitude (m/s²)'); axes[0].set_title(raw_title); axes[0].grid(True, alpha=0.3)

    # Plot 2: Filtered CLEAN accel
    axes[1].plot(time_axis, filtered_accel, 'r-', linewidth=0.6)
    axes[1].axhline( RMS_ACCEL_SIGNIFICANT, color='orange', linestyle='--', alpha=0.6, label=f'Significant: {RMS_ACCEL_SIGNIFICANT}')
    axes[1].axhline( RMS_ACCEL_MILD,        color='gold',   linestyle='--', alpha=0.6, label=f'Mild: {RMS_ACCEL_MILD}')
    axes[1].axhline(-RMS_ACCEL_SIGNIFICANT, color='orange', linestyle='--', alpha=0.6)
    axes[1].axhline(-RMS_ACCEL_MILD,        color='gold',   linestyle='--', alpha=0.6)
    axes[1].set_ylabel('Filtered (m/s²)')
    axes[1].set_title(f'Action Tremor Accel (Bandpass 1-15 Hz, artifact-cleaned)  |  RMS: {tremor_rms_accel:.3f} m/s²')
    axes[1].legend(loc='upper right', fontsize=8); axes[1].grid(True, alpha=0.3)

    # Plot 3: Raw gyro
    axes[2].plot(time_axis, gyro_magnitude_raw, 'g-', linewidth=0.6)
    axes[2].set_ylabel('Magnitude (deg/s)'); axes[2].set_title('Raw Gyroscope Magnitude'); axes[2].grid(True, alpha=0.3)

    # Plot 4: Filtered CLEAN gyro
    axes[3].plot(time_axis, filtered_gyro, 'm-', linewidth=0.6)
    axes[3].axhline( RMS_GYRO_SIGNIFICANT, color='orange', linestyle='--', alpha=0.6, label=f'Significant: {RMS_GYRO_SIGNIFICANT}')
    axes[3].axhline( RMS_GYRO_MILD,        color='gold',   linestyle='--', alpha=0.6, label=f'Mild: {RMS_GYRO_MILD}')
    axes[3].axhline(-RMS_GYRO_SIGNIFICANT, color='orange', linestyle='--', alpha=0.6)
    axes[3].axhline(-RMS_GYRO_MILD,        color='gold',   linestyle='--', alpha=0.6)
    axes[3].set_ylabel('Filtered (deg/s)')
    axes[3].set_title(f'Action Tremor Gyro (Bandpass 1-15 Hz, artifact-cleaned)  |  RMS: {tremor_rms_gyro:.3f} deg/s')
    axes[3].legend(loc='upper right', fontsize=8); axes[3].grid(True, alpha=0.3)

    # Plot 5: Accel PSD (from clean data)
    axes[4].semilogy(freqs_accel, psd_accel, 'b-', linewidth=1.5)
    axes[4].axvspan(4, 6, alpha=0.3, color='red', label='Tremor band (4-6 Hz)')
    axes[4].axvline(peak_freq_accel, color='blue', linestyle='--', label=f'Peak: {peak_freq_accel:.2f} Hz')
    axes[4].set_xlabel('Frequency (Hz)'); axes[4].set_ylabel('PSD'); axes[4].set_xlim(0, 15)
    axes[4].set_title(f'Accelerometer PSD  (Tremor Ratio: {tremor_ratio_accel:.3f})  [bandpass 1-15 Hz, artifact-cleaned]')
    axes[4].legend(); axes[4].grid(True, alpha=0.3)

    # Plot 6: Gyro PSD (from clean data)
    axes[5].semilogy(freqs_gyro, psd_gyro, 'g-', linewidth=1.5)
    axes[5].axvspan(4, 6, alpha=0.3, color='red', label='Tremor band (4-6 Hz)')
    axes[5].axvline(peak_freq_gyro, color='green', linestyle='--', label=f'Peak: {peak_freq_gyro:.2f} Hz')
    axes[5].set_xlabel('Frequency (Hz)'); axes[5].set_ylabel('PSD'); axes[5].set_xlim(0, 15)
    axes[5].set_title(f'Gyroscope PSD  (Tremor Ratio: {tremor_ratio_gyro:.3f})  [bandpass 1-15 Hz, artifact-cleaned]')
    axes[5].legend(); axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_folder, f"{basename}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Text analysis
    report_path = os.path.join(out_folder, f"{basename}_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 55 + "\n")
        f.write("SPIRAL DRAWING TREMOR ANALYSIS REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Session ID:    {session_id}\n")
        f.write(f"Age Group:     {AGE_GROUP}\n")
        f.write(f"Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Spiral Start:  {spiral_start}\n")
        f.write(f"Spiral End:    {spiral_end}\n")
        f.write(f"Duration:      {duration:.1f}s\n")
        f.write(f"Sample Rate:   {SAMPLE_RATE} Hz\n")
        f.write(f"Samples:       {len(ax_data)}\n")
        f.write(f"Filter:        High-pass 2 Hz (Action Tremor)\n\n")
        f.write("THRESHOLDS:\n")
        f.write(f"  Significant:  Ratio > {RATIO_SIGNIFICANT}  |  RMS Accel > {RMS_ACCEL_SIGNIFICANT} m/s²  |  RMS Gyro > {RMS_GYRO_SIGNIFICANT} deg/s\n")
        f.write(f"  Mild:         Ratio > {RATIO_MILD}  |  RMS Accel > {RMS_ACCEL_MILD} m/s²  |  RMS Gyro > {RMS_GYRO_MILD} deg/s\n\n")
        if has_artifacts:
            f.write(f"ARTIFACTS: {artifact_note}\n")
            f.write(f"  {n_accel_artifacts} accel + {n_gyro_artifacts} gyro samples interpolated before analysis.\n\n")
        f.write("ACTION TREMOR - LINEAR (Accelerometer):\n")
        f.write(f"  Peak frequency:  {peak_freq_accel:.2f} Hz\n")
        f.write(f"  Tremor ratio:    {tremor_ratio_accel:.3f}\n")
        f.write(f"  RMS amplitude:   {tremor_rms_accel:.3f} m/s²\n")
        f.write(f"  Peak-to-peak:    {tremor_p2p_accel:.3f} m/s²\n\n")
        f.write("ACTION TREMOR - ROTATIONAL (Gyroscope):\n")
        f.write(f"  Peak frequency:  {peak_freq_gyro:.2f} Hz\n")
        f.write(f"  Tremor ratio:    {tremor_ratio_gyro:.3f}\n")
        f.write(f"  RMS amplitude:   {tremor_rms_gyro:.3f} deg/s\n")
        f.write(f"  Peak-to-peak:    {tremor_p2p_gyro:.3f} deg/s\n\n")
        f.write("DRAWING QUALITY:\n")
        f.write(f"  Smoothness score: {smoothness_score:.3f}  (lower = smoother)\n\n")
        f.write("OVERALL ASSESSMENT:\n")
        f.write(f"  Status: {tremor_status}\n")
        f.write(f"  Peak prominent (accel): {accel_peak_prominent}\n")
        f.write(f"  Peak prominent (gyro):  {gyro_peak_prominent}\n")

    print(f"  Saved: {out_folder}")
    return "ok"

# Main
total_success = 0
total_skipped = 0

for AGE_GROUP in AGE_GROUPS:
    AGE_DIR   = os.path.join(EXPORT_DIR, AGE_GROUP)
    ACCEL_OUT = os.path.join(EXPORT_DIR, "accel_reading", f"accel_{AGE_GROUP}")

    print("\n" + "=" * 55)
    print(f"Spiral Tremor Analysis  |  Age group: {AGE_GROUP}")
    print(f"Processing all testers in {AGE_DIR}")
    print("=" * 55)

    if not os.path.exists(AGE_DIR):
        print(f"  SKIP: Folder not found: {AGE_DIR}")
        continue

    tester_dirs = sorted([
        d for d in os.listdir(AGE_DIR)
        if os.path.isdir(os.path.join(AGE_DIR, d))
    ])

    if len(tester_dirs) == 0:
        print(f"  SKIP: No tester folders found")
        continue

    print(f"Found {len(tester_dirs)} tester(s):\n  " + "\n  ".join(tester_dirs) + "\n")
    os.makedirs(ACCEL_OUT, exist_ok=True)

    success = 0
    skipped = 0
    for session_id in tester_dirs:
        tester_path = os.path.join(AGE_DIR, session_id)
        try:
            result = analyze_tester(session_id, tester_path, ACCEL_OUT)
            if result == "ok":
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR processing {session_id}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1

    print(f"\n  Age group {AGE_GROUP}: {success} completed  |  {skipped} skipped/errored")
    total_success += success
    total_skipped += skipped

print("\n" + "=" * 55)
print(f"ALL GROUPS DONE  |  {total_success} completed  |  {total_skipped} skipped/errored")
print(f"Results in: {os.path.join(EXPORT_DIR, 'accel_reading')}")
print("=" * 55)
