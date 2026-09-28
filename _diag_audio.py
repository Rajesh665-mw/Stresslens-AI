import sounddevice as sd
import numpy as np
import time

# Test device 2 = Intel Microphone Array (built-in laptop mic)
sr = 16000
buf = []
def cb(indata, frames, time_info, status):
    buf.extend(indata[:, 0].tolist())

stream = sd.InputStream(samplerate=sr, channels=1, blocksize=800, dtype='float32', callback=cb, device=2)
stream.start()
print("Recording 3 seconds from DEVICE 2 (Intel Mic Array)... SPEAK NOW!")
time.sleep(3)
stream.stop()
stream.close()

data = np.array(buf, dtype=np.float32)
rms = float(np.sqrt(np.mean(np.square(data))))
peak = float(np.max(np.abs(data)))
print(f"Captured {len(data)} samples ({len(data)/sr:.1f}s)")
print(f"RMS: {rms:.6f}  Peak: {peak:.6f}")
print(f"RMS > 0.01 threshold? {'YES - audio working!' if rms > 0.01 else 'NO - still too quiet'}")
