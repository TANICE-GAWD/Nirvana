import numpy as np
import pyroomacoustics as pra

from onlyyou import proximity, vibration
from onlyyou.audio import HOP, SR


def _render(src_offset, rng, spacing=0.025):
    room_dim = [6, 5, 3]
    absorption, max_order = pra.inverse_sabine(0.4, room_dim)
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(absorption), max_order=10)
    chest = np.array([3.0, 2.5, 1.3])
    half = np.array([0, 0, spacing / 2])
    room.add_microphone_array(np.stack([chest + half, chest - half], axis=1))
    room.add_source(chest + np.array(src_offset), signal=rng.normal(size=SR * 2).astype(np.float32))
    room.simulate()
    return room.mic_array.signals[:, : SR * 2].astype(np.float32)


def test_proximity_separates_mouth_from_far_talker():
    rng = np.random.default_rng(0)
    lag = proximity.wearer_lag(0.025)
    near = proximity.features(_render([0.05, 0, 0.15], rng), lag)[10:]
    far = proximity.features(_render([2.0, 0.5, 0.3], rng), lag)[10:]
    # top mic louder than bottom only for the mouth
    assert near[:, 1].mean() > far[:, 1].mean() + 0.3
    # arrival-time difference matches the wearer's direction for the mouth only
    assert np.median(near[:, 2]) < np.median(far[:, 2])
    # direct sound -> higher inter-mic coherence
    assert near[:, 4].mean() > far[:, 4].mean()


def test_imu_emulation_rates_and_aliasing():
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    voice = np.sin(2 * np.pi * 180 * t).astype(np.float32)  # typical voice pitch
    for rate in (2000, 500, 100):
        y = vibration.emulate_imu(voice, rate, aliased=False, rng=rng, noise_db=-80)
        assert abs(len(y) - rate) <= 1
    # a 100 Hz IMU with an anti-aliasing filter removes 180 Hz voice energy; without it, energy folds in
    filt = vibration.emulate_imu(voice, 100, aliased=False, rng=rng, noise_db=-80)
    alias = vibration.emulate_imu(voice, 100, aliased=True, rng=rng, noise_db=-80)
    assert np.std(alias[5:-5]) > 10 * np.std(filt[5:-5])


def test_vibration_features_shape():
    rng = np.random.default_rng(0)
    imu = rng.normal(size=1000).astype(np.float32)
    mic_db = np.zeros(SR // HOP, np.float32)
    f = vibration.features(imu, 1000, mic_db)
    assert f.shape == (len(mic_db), 3)
