import os
import csv
import re
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from datetime import datetime


EXPORT_DIR  = os.path.join(os.path.expanduser("~"), "Desktop", "export")
AGE_GROUPS  = ["18-24", "41-50", "51-60", "unknow"]


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

def bandpass_filter(data, lowcut=0.7, highcut=3.5, fs=100):
    nyq  = fs / 2
    low  = max(lowcut / nyq,  0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, data)

def calc_hr(ir_array, fs):
    if len(ir_array) < fs * 5:
        return None
    filtered = bandpass_filter(ir_array, fs=fs)
    min_dist = int(fs * 0.4)
    peaks, _ = find_peaks(filtered, distance=min_dist,
                           prominence=np.std(filtered) * 0.3)
    if len(peaks) < 2:
        return None
    hr = 60 / (np.median(np.diff(peaks)) / fs)
    return round(float(hr), 1) if 40 < hr < 180 else None

def calc_spo2(ir_array, red_array):
    ir  = np.array(ir_array,  dtype=float)
    red = np.array(red_array, dtype=float)
    ir_ac  = np.std(ir);   ir_dc  = np.mean(ir)
    red_ac = np.std(red);  red_dc = np.mean(red)
    if ir_dc == 0 or red_dc == 0 or ir_ac == 0:
        return None
    r    = (red_ac / red_dc) / (ir_ac / ir_dc)
    spo2 = 110 - 25 * r
    return round(float(spo2), 1) if 80 <= spo2 <= 100 else None

def fmt_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


# Per-tester analysis
def analyze_tester(session_id, tester_path, out_base):
    print("\n" + "=" * 55)
    print(f"Session: {session_id}")
    print("=" * 55)

    # Find sensor_timeseries.csv
    sensor_csv = os.path.join(tester_path, "sensor_timeseries.csv")
    if not os.path.exists(sensor_csv):
        for f in os.listdir(tester_path):
            if f.lower() == "sensor_timeseries.csv":
                sensor_csv = os.path.join(tester_path, f)
                break
        else:
            print(f"  SKIP: sensor_timeseries.csv not found")
            print(f"  Files: {os.listdir(tester_path)}")
            return "skipped"


    # Load oxi rows — full recording
    rows     = []
    ts_list  = []

    with open(sensor_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}

            ts = parse_ts(row.get('timestamp', ''))
            if ts is None:
                continue

            oxi_valid = safe_bool(row.get('ox_is_valid', 'false'))
            red = safe_float(row.get('ox_red_signal'))
            ir  = safe_float(row.get('ox_infrared_signal'))

            if oxi_valid and red is not None and ir is not None:
                rows.append({'red': red, 'ir': ir})
                ts_list.append(ts)

    if len(rows) == 0:
        print(f"  SKIP: No valid oxi data found")
        return "skipped"

    # Sort ascending
    if len(ts_list) >= 2:
        sorted_pairs = sorted(zip(ts_list, rows), key=lambda x: x[0])
        ts_list = [p[0] for p in sorted_pairs]
        rows    = [p[1] for p in sorted_pairs]

    # Auto-detect sample rate
    if len(ts_list) >= 2:
        total_secs  = (ts_list[-1] - ts_list[0]).total_seconds()
        SAMPLE_RATE = int(round(len(rows) / total_secs)) if total_secs > 0 else 100
    else:
        SAMPLE_RATE = 100

    total_duration = len(rows) / SAMPLE_RATE
    print(f"  Oxi samples: {len(rows)}  |  Rate: {SAMPLE_RATE} Hz  |  Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

    ir_data  = np.array([r['ir']  for r in rows], dtype=float)
    red_data = np.array([r['red'] for r in rows], dtype=float)

    if np.mean(ir_data) < 50000 or np.mean(red_data) < 50000:
        print(f"  WARNING: Low signal — finger may not have been on sensor throughout")


    # Second-by-second timeline using 10s lookback window
    LOOKBACK = SAMPLE_RATE * 10
    HALF     = LOOKBACK // 2

    timeline = []

    for sec in range(1, int(total_duration) + 1):
        centre  = sec * SAMPLE_RATE
        start   = max(0, centre - HALF)
        end     = min(len(ir_data), centre + HALF)
        ir_seg  = ir_data[start:end]
        red_seg = red_data[start:end]
        time_str = fmt_time(sec)
        note     = ""

        if np.mean(ir_seg) < 50000 or np.mean(red_seg) < 50000:
            hr_out = "---"; spo2_out = "---"; note = "no finger"
        elif len(ir_seg) < SAMPLE_RATE * 5:
            hr_out = "---"; spo2_out = "---"; note = "insufficient data"
        else:
            hr   = calc_hr(ir_seg, SAMPLE_RATE)
            spo2 = calc_spo2(ir_seg, red_seg)
            hr_out   = f"{hr:.0f} bpm" if hr   is not None else "---"
            spo2_out = f"{spo2:.1f}%"  if spo2 is not None else "---"

        timeline.append({'time': time_str, 'hr': hr_out, 'spo2': spo2_out, 'note': note})


    # Overall summary
    valid_hrs   = []
    valid_spo2s = []
    for entry in timeline:
        if entry['note'] == '':
            if entry['hr'] not in ('---', ''):
                try:
                    valid_hrs.append(float(entry['hr'].replace(' bpm', '')))
                except: pass
            if entry['spo2'] not in ('---', ''):
                try:
                    valid_spo2s.append(float(entry['spo2'].replace('%', '')))
                except: pass

    final_hr   = round(float(np.median(valid_hrs)),   1) if valid_hrs   else None
    final_spo2 = round(float(np.median(valid_spo2s)), 1) if valid_spo2s else None
    hr_std     = round(float(np.std(valid_hrs)),   1) if len(valid_hrs)   > 1 else None
    spo2_std   = round(float(np.std(valid_spo2s)), 1) if len(valid_spo2s) > 1 else None

    hr_summary   = f"{final_hr:.0f} bpm  (std: {hr_std} bpm, {len(valid_hrs)} readings)"   if final_hr   else "Could not calculate"
    spo2_summary = f"{final_spo2:.1f}%   (std: {spo2_std}%, {len(valid_spo2s)} readings)"  if final_spo2 else "Could not calculate"

    print(f"  Heart Rate:   {hr_summary}")
    print(f"  SpO2:         {spo2_summary}")

    # Output folder

    out_folder = os.path.join(out_base, f"{session_id}_hr_spo2_results")
    os.makedirs(out_folder, exist_ok=True)
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(out_folder, f"{session_id}_{run_ts}_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 55 + "\n")
        f.write("HR & SpO2 ANALYSIS REPORT  (Full Recording)\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Session ID:    {session_id}\n")
        f.write(f"Age Group:     {AGE_GROUP}\n")
        f.write(f"Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Rate:   {SAMPLE_RATE} Hz\n")
        f.write(f"Duration:      {total_duration:.1f}s ({total_duration/60:.1f} min)\n\n")

        f.write("=" * 55 + "\n")
        f.write("SECOND-BY-SECOND TIMELINE\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'Time':>6}  {'HR':>8}  {'SpO2':>8}  {'Note'}\n")
        f.write("-" * 45 + "\n")
        for entry in timeline:
            note_str = f"  [{entry['note']}]" if entry['note'] else ""
            f.write(f"{entry['time']:>6}  {entry['hr']:>8}  {entry['spo2']:>8}{note_str}\n")
        f.write("-" * 45 + "\n\n")

        f.write("=" * 55 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 55 + "\n")
        f.write(f"Heart Rate:    {hr_summary}\n")
        if valid_hrs:
            f.write(f"  HR range:    {min(valid_hrs):.0f} - {max(valid_hrs):.0f} bpm\n")
        f.write(f"Blood Oxygen:  {spo2_summary}\n")
        if valid_spo2s:
            f.write(f"  SpO2 range:  {min(valid_spo2s):.1f} - {max(valid_spo2s):.1f}%\n")

    print(f"  Saved: {out_folder}")
    return "ok"

# Main
total_success = 0
total_skipped = 0

for AGE_GROUP in AGE_GROUPS:
    AGE_DIR = os.path.join(EXPORT_DIR, AGE_GROUP)
    OXI_OUT = os.path.join(EXPORT_DIR, "oxi_reading", f"oxi_{AGE_GROUP}")

    print("\n" + "=" * 55)
    print(f"HR & SpO2 Analysis  |  Age group: {AGE_GROUP}")
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
    os.makedirs(OXI_OUT, exist_ok=True)

    success = 0
    skipped = 0
    for session_id in tester_dirs:
        tester_path = os.path.join(AGE_DIR, session_id)
        try:
            result = analyze_tester(session_id, tester_path, OXI_OUT)
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
print(f"Results in: {os.path.join(EXPORT_DIR, 'oxi_reading')}")
print("=" * 55)
