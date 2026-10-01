"""Small sounddevice helpers: pick devices by name, record several mics at once."""

import threading
import time

import numpy as np
import sounddevice as sd

from onlyyou.audio import SR, resample


def find(name, kind="input"):
    """Device index whose name contains `name` (case-insensitive); None -> system default."""
    if name is None:
        return None
    if str(name).isdigit():
        return int(name)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d[key] > 0:
            return i
    raise SystemExit(f"No {kind} device matching '{name}'. Run with `devices` to list them.")


def list_devices():
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:3d}  in={d['max_input_channels']:<3d} out={d['max_output_channels']:<3d} "
              f"{int(d['default_samplerate'])} Hz  {d['name']}")


class MultiRecorder:
    """Record from several input devices at the same time; returns 16 kHz mono arrays."""

    def __init__(self, devices: dict):
        self.devices = devices
        self.buffers = {k: [] for k in devices}
        self.rates, self.streams = {}, []

    def __enter__(self):
        for tag, dev in self.devices.items():
            info = sd.query_devices(dev, "input")
            rate = int(info["default_samplerate"])
            self.rates[tag] = rate

            def cb(indata, frames, t, status, tag=tag):
                self.buffers[tag].append(indata[:, 0].copy())

            s = sd.InputStream(device=dev, channels=1, samplerate=rate, callback=cb, dtype="float32")
            s.start()
            self.streams.append(s)
        return self

    def __exit__(self, *exc):
        for s in self.streams:
            s.stop()
            s.close()

    def result(self):
        return {k: resample(np.concatenate(v) if v else np.zeros(0, np.float32), self.rates[k], SR)
                for k, v in self.buffers.items()}


def countdown(seconds, label):
    stop = threading.Event()

    def run():
        for left in range(seconds, 0, -1):
            if stop.is_set():
                return
            print(f"\r{label}: {left:3d}s left ", end="", flush=True)
            time.sleep(1)
        print()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return stop, t
