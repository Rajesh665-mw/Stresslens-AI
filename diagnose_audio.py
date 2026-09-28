"""
Diagnostic script for testing microphone input and audio RMS signal levels.
Lists available audio input devices and probes recording quality.
"""
import argparse
import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError as e:
    print(f"Error: sounddevice is not installed: {e}")
    print("Run: pip install -r requirements-perception.txt")
    sys.exit(1)


def list_devices():
    print("\n--- Available Audio Input Devices ---")
    devices = sd.query_devices()
    input_devices = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            input_devices.append((idx, d["name"]))
            marker = " [DEFAULT]" if idx == sd.default.device[0] else ""
            print(f"  [{idx}] {d['name']} ({d['max_input_channels']} in){marker}")
    if not input_devices:
        print("  No audio input devices detected.")
    print("------------------------------------\n")
    return input_devices


def main():
    parser = argparse.ArgumentParser(description="Test microphone input and signal RMS level.")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (defaults to system default)")
    parser.add_argument("--duration", type=float, default=3.0, help="Test recording duration in seconds")
    parser.add_argument("--list", action="store_true", help="List audio input devices and exit")
    args = parser.parse_args()

    input_devices = list_devices()
    if args.list:
        return

    device_index = args.device
    if device_index is None:
        try:
            device_index = sd.default.device[0]
            if device_index < 0:
                device_index = input_devices[0][0] if input_devices else 0
        except Exception:
            device_index = 0

    sr = 16000
    buf = []

    def audio_callback(indata, frames, time_info, status):
        buf.extend(indata[:, 0].tolist())

    print(f"Recording {args.duration:.1f}s from device index {device_index}... SPEAK INTO MICROPHONE NOW!")
    try:
        stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            blocksize=800,
            dtype="float32",
            callback=audio_callback,
            device=device_index,
        )
        stream.start()
        time.sleep(args.duration)
        stream.stop()
        stream.close()
    except Exception as err:
        print(f"Failed to record from device {device_index}: {err}")
        sys.exit(1)

    data = np.array(buf, dtype=np.float32)
    if len(data) == 0:
        print("No samples captured.")
        return

    rms = float(np.sqrt(np.mean(np.square(data))))
    peak = float(np.max(np.abs(data)))
    print(f"\nCaptured {len(data)} samples ({len(data)/sr:.1f}s)")
    print(f"RMS:  {rms:.6f}")
    print(f"Peak: {peak:.6f}")
    print(f"Signal Status: {'PASS - Audio working!' if rms > 0.008 else 'WARNING - Audio too quiet / silent'}\n")


if __name__ == "__main__":
    main()
