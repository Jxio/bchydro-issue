import sounddevice as sd
import numpy as np
import time
import csv
from datetime import datetime

# ============================================================
# Settings
# ============================================================

SAMPLE_RATE = 48000
FFT_SIZE = 32768

# How long to measure each location
MEASURE_SECONDS = 60

# How often to generate one measurement
REPORT_INTERVAL = 1.0

# Frequencies of interest
TARGETS = [29, 58, 116, 120, 174, 232, 240]

# Frequency range for finding actual peaks
MIN_FREQ = 20
MAX_FREQ = 400

# ============================================================
# Global recording buffer
# ============================================================

audio_buffer = np.zeros(FFT_SIZE, dtype=np.float32)


# ============================================================
# Audio callback
# ============================================================

def audio_callback(indata, frames, time_info, status):
    global audio_buffer

    if status:
        print(f"⚠️ {status}")

    samples = indata[:, 0]

    audio_buffer = np.roll(
        audio_buffer,
        -len(samples)
    )

    audio_buffer[-len(samples):] = samples


# ============================================================
# Spectrum calculation
# ============================================================

def calculate_spectrum(data):

    data = data.astype(np.float64)

    # Remove DC offset
    data -= np.mean(data)

    # Hann window
    window = np.hanning(len(data))

    windowed = data * window

    # FFT
    spectrum = np.abs(
        np.fft.rfft(windowed)
    )

    freqs = np.fft.rfftfreq(
        len(data),
        1 / SAMPLE_RATE
    )

    return freqs, spectrum


# ============================================================
# Measure target frequency
# ============================================================

def get_target_level(freqs, spectrum, target):

    # Search ±2 Hz around target
    mask = (
        (freqs >= target - 2)
        &
        (freqs <= target + 2)
    )

    if not np.any(mask):
        return 0.0

    return float(
        np.max(spectrum[mask])
    )


# ============================================================
# Find strongest peaks
# ============================================================

def find_peaks(freqs, spectrum):

    mask = (
        (freqs >= MIN_FREQ)
        &
        (freqs <= MAX_FREQ)
    )

    f = freqs[mask]
    s = spectrum[mask]

    # Ignore extremely small values
    if len(s) == 0:
        return []

    # Find local maxima
    local_peaks = []

    for i in range(1, len(s) - 1):

        if s[i] > s[i - 1] and s[i] > s[i + 1]:

            local_peaks.append(i)

    # Sort by amplitude
    local_peaks.sort(
        key=lambda i: s[i],
        reverse=True
    )

    # Keep peaks separated by at least ~5 Hz
    selected = []

    for i in local_peaks:

        frequency = f[i]

        if all(
            abs(frequency - f[j]) >= 5
            for j in selected
        ):
            selected.append(i)

        if len(selected) >= 5:
            break

    return [
        (float(f[i]), float(s[i]))
        for i in selected
    ]


# ============================================================
# Measure one location
# ============================================================

def measure_location(location):

    global audio_buffer

    print()
    print("==============================================")
    print(f"📍 LOCATION: {location.upper()}")
    print("==============================================")
    print()
    print(
        f"Measuring for {MEASURE_SECONDS} seconds."
    )
    print(
        "Keep the Mac and microphone completely stationary."
    )
    print()

    # Give the microphone a moment to stabilize
    time.sleep(2)

    records = []

    start_time = time.time()
    next_report = start_time

    while True:

        now = time.time()

        elapsed = now - start_time

        if elapsed >= MEASURE_SECONDS:
            break

        if now < next_report:
            time.sleep(0.02)
            continue

        next_report += REPORT_INTERVAL

        data = audio_buffer.copy()

        freqs, spectrum = calculate_spectrum(data)

        levels = {}

        for target in TARGETS:

            levels[target] = get_target_level(
                freqs,
                spectrum,
                target
            )

        peaks = find_peaks(
            freqs,
            spectrum
        )

        record = {
            "location": location,
            "elapsed": elapsed,
        }

        for target in TARGETS:

            record[
                f"{target}Hz"
            ] = levels[target]

        for i in range(5):

            if i < len(peaks):

                record[
                    f"peak{i+1}_freq"
                ] = peaks[i][0]

                record[
                    f"peak{i+1}_level"
                ] = peaks[i][1]

            else:

                record[
                    f"peak{i+1}_freq"
                ] = ""

                record[
                    f"peak{i+1}_level"
                ] = ""

        records.append(record)

        remaining = MEASURE_SECONDS - elapsed

        peak_text = ""

        if peaks:

            peak_text = " | ".join(
                f"{freq:.1f}Hz={level:.6f}"
                for freq, level in peaks[:3]
            )

        print(
            f"[{elapsed:5.1f}s] "
            f"58={levels[58]:.6f}  "
            f"116={levels[116]:.6f}  "
            f"120={levels[120]:.6f}  "
            f"240={levels[240]:.6f}"
        )

        print(
            f"         peaks: {peak_text}"
        )

    print()
    print(
        f"✅ {location.upper()} measurement complete."
    )

    return records


# ============================================================
# Summary
# ============================================================

def print_summary(all_records):

    print()
    print()
    print("======================================================")
    print("                 LOCATION SUMMARY")
    print("======================================================")

    locations = [
        "bed",
        "window",
        "washroom",
        "elec room"
    ]

    for location in locations:

        rows = [
            r for r in all_records
            if r["location"] == location
        ]

        if not rows:
            continue

        print()
        print(
            f"📍 {location.upper()} "
            f"({len(rows)} samples)"
        )

        print(
            "------------------------------------------------------"
        )

        for target in TARGETS:

            values = np.array([
                r[f"{target}Hz"]
                for r in rows
            ])

            print(
                f"{target:3} Hz   "
                f"avg={np.mean(values):.6f}   "
                f"median={np.median(values):.6f}   "
                f"max={np.max(values):.6f}"
            )

    # ========================================================
    # Relative comparison
    # ========================================================

    print()
    print()
    print("======================================================")
    print("              RELATIVE LOCATION COMPARISON")
    print("======================================================")

    for target in [29, 58, 116, 120, 174, 232, 240]:

        medians = {}

        for location in locations:

            rows = [
                r for r in all_records
                if r["location"] == location
            ]

            if rows:

                values = np.array([
                    r[f"{target}Hz"]
                    for r in rows
                ])

                medians[location] = np.median(values)

        if not medians:
            continue

        strongest = max(
            medians,
            key=medians.get
        )

        print()

        print(
            f"{target:3} Hz:"
        )

        for location, value in medians.items():

            print(
                f"   {location:7} "
                f"{value:.6f}"
            )

        print(
            f"   → strongest: {strongest.upper()}"
        )


# ============================================================
# Save CSV
# ============================================================

def save_csv(records):

    if not records:
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"location_mapping_3_3_1_"
        f"{timestamp}.csv"
    )

    fieldnames = list(records[0].keys())

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    print()
    print(
        f"💾 CSV saved: {filename}"
    )


# ============================================================
# Main
# ============================================================

def main():

    global audio_buffer

    print()
    print("======================================================")
    print("        V3.3.1 LOW-FREQUENCY LOCATION MAPPING")
    print("======================================================")
    print()
    print("This version measures one location at a time.")
    print()
    print("Important:")
    print("  • Keep the Mac stationary.")
    print("  • Put the microphone near your head position.")
    print("  • Do not walk around during measurement.")
    print("  • Measure each location for 60 seconds.")
    print()

    all_records = []

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=audio_callback
        ):

            while True:

                print()
                print("------------------------------------------------------")
                print("Select location:")
                print()
                print("  1 = BED")
                print("  2 = WINDOW")
                print("  3 = washroom")
                print("  4 = elec room")
                print("  q = QUIT")
                print()

                choice = input(
                    "Enter: "
                ).strip().lower()

                if choice == "q":
                    break

                if choice == "1":
                    location = "bed"

                elif choice == "2":
                    location = "window"

                elif choice == "3":
                    location = "washroom"
                
                elif choice == "4":
                    location = "elec room"
                else:
                    print(
                        "❌ Invalid choice."
                    )
                    continue

                records = measure_location(
                    location
                )

                all_records.extend(
                    records
                )

                print()
                print(
                    "You can now move the Mac "
                    "to another location."
                )

    except KeyboardInterrupt:

        print()
        print("Stopped by Ctrl+C.")

    except Exception as e:

        print()
        print(
            f"❌ Error: {e}"
        )

    # ========================================================
    # Final output
    # ========================================================

    if all_records:

        print_summary(
            all_records
        )

        save_csv(
            all_records
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
