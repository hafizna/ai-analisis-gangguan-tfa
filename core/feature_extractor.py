"""
Ekstraktor Fitur Rele Jarak
============================
Mengekstrak fitur berdasarkan jenis proteksi (rele jarak / rele diferensial).

Fitur rele jarak : impedans (R/X, besar Z, tegangan jatuh/sag)
Fitur rele diferensial : arus saja (untuk pengembangan selanjutnya)
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import logging
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


@dataclass
class DistanceFeatures:
    """Features extracted from distance relay recordings."""

    # === IMPEDANCE FEATURES (primary for distance) ===
    r_x_ratio: Optional[float]           # R/X from fault impedance
    z_magnitude_ohms: Optional[float]     # |Z| in primary ohms
    z_angle_degrees: Optional[float]      # Impedance angle
    voltage_sag_depth_pu: float           # Min voltage during fault (per-unit)
    voltage_sag_phase: str                # Which phase had deepest sag

    # === CURRENT FEATURES (universal) ===
    di_dt_max: float                      # Max |dI/dt| in first cycle after inception (A/s)
    di_dt_phase: str                      # Phase with highest dI/dt
    peak_fault_current_a: float           # Maximum instantaneous current (Amps primary)
    peak_fault_phase: str                 # Phase with highest peak
    i0_i1_ratio: float                    # Zero-sequence / positive-sequence current ratio
    thd_percent: float                    # THD of fault current, first 2 cycles

    # === INCEPTION FEATURES ===
    inception_angle_degrees: float        # Voltage phase angle at fault inception (0-360)

    # === PROTECTION CONTEXT (free features from status channels) ===
    zone_operated: str                    # "Z1", "Z2", "Z3", or "Z1+Z2" etc
    teleprotection_received: bool         # Was permission signal received?
    comms_failure: bool                   # Communication link failure detected?
    trip_type: str                        # "single_pole" or "three_pole"

    # === RECLOSE FEATURES (very strong for lightning) ===
    reclose_attempted: bool
    reclose_successful: Optional[bool]    # True = transient fault (likely lightning)
    reclose_time_ms: Optional[float]      # Dead time before reclose
    fault_count: int                      # 1 = single fault, 2+ = reclose failed

    # === FAULT TYPE (supporting) ===
    faulted_phases: List[str]             # ["A"], ["A","B"], etc
    fault_type: str                       # "SLG", "DLG", "LL", "3PH"
    is_ground_fault: bool                 # True if ground/neutral involved

    # === METADATA ===
    station_name: str
    relay_model: str
    voltage_kv: Optional[float]
    sampling_rate_hz: float
    record_duration_ms: float

    # === SYMMETRICAL COMPONENT MAGNITUDES (RMS, primary amps) ===
    i0_magnitude_a: Optional[float] = None
    i1_magnitude_a: Optional[float] = None
    i2_magnitude_a: Optional[float] = None

    # === VOLTAGE LEVELS (primary volts, phase-to-ground RMS) ===
    v_prefault_v: Optional[float] = None
    v_fault_v: Optional[float] = None


@dataclass
class DifferentialFeatures:
    """Features extracted from differential relay recordings.
    NOT used for classification yet — stored for future use."""

    # Universal current features (same as distance)
    di_dt_max: float
    di_dt_phase: str
    peak_fault_current_a: float
    i0_i1_ratio: float
    thd_percent: float
    inception_angle_degrees: float

    # Differential-specific (for future classifier)
    idiff_max_percent: Optional[float]    # Max differential current (%)
    irestraint_max_percent: Optional[float]  # Max restraint current (%)
    idiff_rise_rate: Optional[float]      # dIdiff/dt

    # Reclose (if available)
    reclose_attempted: bool
    reclose_successful: Optional[bool]

    # Fault type
    faulted_phases: List[str]
    fault_type: str
    is_ground_fault: bool

    # Metadata
    station_name: str
    relay_model: str
    voltage_kv: Optional[float]
    sampling_rate_hz: float

    # Flag
    classification_status: str = "UNCLASSIFIED - awaiting differential analysis module"


def extract_distance_features(record, fault, protection) -> Optional[DistanceFeatures]:
    """
    Extract features for distance protection recordings.

    Args:
        record: ComtradeRecord
        fault: FaultEvent from fault_detector
        protection: ProtectionEvent from protection_router

    Returns:
        DistanceFeatures or None if extraction fails
    """

    try:
        # Get phase channels
        ia = _get_channel(record, 'IA', 'current')
        ib = _get_channel(record, 'IB', 'current')
        ic = _get_channel(record, 'IC', 'current')
        va = _get_channel(record, 'VA', 'voltage')
        vb = _get_channel(record, 'VB', 'voltage')
        vc = _get_channel(record, 'VC', 'voltage')

        if not all([ia, ib, ic, va, vb, vc]):
            logger.error("Missing required phase channels for distance feature extraction")
            return None

        # Calculate sampling rate
        sampling_rate = _get_sampling_rate(record)
        if sampling_rate == 0:
            logger.error("Cannot calculate sampling rate")
            return None

        inception_idx = fault.inception_idx
        system_freq = record.frequency if record.frequency else 50.0

        # Calculate impedance features
        r_x_ratio, z_mag, z_angle = _calculate_impedance(
            va, vb, vc, ia, ib, ic, inception_idx, sampling_rate, system_freq, fault.faulted_phases
        )

        # Calculate voltage sag
        voltage_sag_depth, voltage_sag_phase = _calculate_voltage_sag(
            va, vb, vc, inception_idx, sampling_rate, system_freq
        )

        # Calculate dI/dt features
        di_dt_max, di_dt_phase = _calculate_di_dt(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )

        # Calculate peak fault current
        peak_current, peak_phase = _calculate_peak_current(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )

        # Calculate symmetrical components (ratio + magnitudes)
        i0_i1_ratio = _calculate_i0_i1_ratio(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )
        i0_mag, i1_mag, i2_mag = _calculate_symmetrical_magnitudes(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )

        # Calculate THD
        thd_percent = _calculate_thd(
            ia, ib, ic, inception_idx, sampling_rate, system_freq, fault.faulted_phases
        )

        # Calculate inception angle
        inception_angle = _calculate_inception_angle(
            va, vb, vc, inception_idx, sampling_rate, system_freq, fault.faulted_phases
        )

        # Determine fault type
        fault_type, is_ground_fault = _determine_fault_type(fault.faulted_phases, i0_i1_ratio)

        # Get voltage level
        voltage_kv = _estimate_voltage_level(va, vb, vc)

        # Get absolute voltage levels (prefault vs fault)
        v_prefault, v_fault = _calculate_voltage_levels(
            va, vb, vc, inception_idx, sampling_rate, system_freq, voltage_sag_phase
        )

        # Build feature object
        return DistanceFeatures(
            # Impedance
            r_x_ratio=r_x_ratio,
            z_magnitude_ohms=z_mag,
            z_angle_degrees=z_angle,
            voltage_sag_depth_pu=voltage_sag_depth,
            voltage_sag_phase=voltage_sag_phase,
            # Current
            di_dt_max=di_dt_max,
            di_dt_phase=di_dt_phase,
            peak_fault_current_a=peak_current,
            peak_fault_phase=peak_phase,
            i0_i1_ratio=i0_i1_ratio,
            thd_percent=thd_percent,
            # Inception
            inception_angle_degrees=inception_angle,
            # Protection context
            zone_operated='+'.join(protection.operated_zones) if protection.operated_zones else 'UNKNOWN',
            teleprotection_received=protection.permission_received,
            comms_failure=protection.comms_failure,
            trip_type=protection.trip_type,
            # Reclose — prefer protection router result; fall back to fault detector
            # reclose_events for dead-time recordings where the router has no AR signals.
            reclose_attempted=(
                protection.auto_reclose_attempted or bool(fault.reclose_events)
            ),
            reclose_successful=(
                protection.auto_reclose_successful
                if protection.auto_reclose_successful is not None
                else (fault.reclose_events[0]['success'] if fault.reclose_events else None)
            ),
            reclose_time_ms=fault.reclose_events[0]['time'] * 1000 if fault.reclose_events else None,
            fault_count=len(fault.reclose_events) + 1,
            # Fault type
            faulted_phases=fault.faulted_phases,
            fault_type=fault_type,
            is_ground_fault=is_ground_fault,
            # Metadata
            station_name=record.station_name,
            relay_model=record.rec_dev_id,
            voltage_kv=voltage_kv,
            sampling_rate_hz=sampling_rate,
            record_duration_ms=record.time[-1] * 1000 if len(record.time) > 0 else 0.0,
            # Extended electrical fields
            i0_magnitude_a=i0_mag,
            i1_magnitude_a=i1_mag,
            i2_magnitude_a=i2_mag,
            v_prefault_v=v_prefault,
            v_fault_v=v_fault,
        )

    except Exception as e:
        logger.error(f"Failed to extract distance features: {e}", exc_info=True)
        return None


def extract_differential_features(record, fault, protection) -> Optional[DifferentialFeatures]:
    """
    Extract universal features from differential recordings.
    These are stored but NOT used for classification yet.
    """

    try:
        # Get phase channels
        ia = _get_channel(record, 'IA', 'current')
        ib = _get_channel(record, 'IB', 'current')
        ic = _get_channel(record, 'IC', 'current')
        va = _get_channel(record, 'VA', 'voltage')
        vb = _get_channel(record, 'VB', 'voltage')
        vc = _get_channel(record, 'VC', 'voltage')

        if not all([ia, ib, ic]):
            logger.error("Missing required current channels")
            return None

        sampling_rate = _get_sampling_rate(record)
        if sampling_rate == 0:
            return None

        inception_idx = fault.inception_idx
        system_freq = record.frequency if record.frequency else 50.0

        # Universal features
        di_dt_max, di_dt_phase = _calculate_di_dt(ia, ib, ic, inception_idx, sampling_rate, system_freq)
        peak_current, _ = _calculate_peak_current(ia, ib, ic, inception_idx, sampling_rate, system_freq)
        i0_i1_ratio = _calculate_i0_i1_ratio(ia, ib, ic, inception_idx, sampling_rate, system_freq)
        thd_percent = _calculate_thd(ia, ib, ic, inception_idx, sampling_rate, system_freq, fault.faulted_phases)
        inception_angle = _calculate_inception_angle(va, vb, vc, inception_idx, sampling_rate, system_freq, fault.faulted_phases) if va else 0.0

        # Differential-specific features (if channels available)
        idiff_max = None
        irestraint_max = None
        idiff_rise_rate = None

        # Look for differential channels
        idiff_a = _get_channel(record, 'Ln1:87L:I-DIFF:I diff.:phs A', None)
        if idiff_a and len(idiff_a.samples) > inception_idx:
            # Calculate max differential current
            window = slice(inception_idx, min(inception_idx + int(2*sampling_rate/system_freq), len(idiff_a.samples)))
            idiff_max = np.max(np.abs(idiff_a.samples[window]))

        # Fault type
        fault_type, is_ground_fault = _determine_fault_type(fault.faulted_phases, i0_i1_ratio)

        # Voltage level
        voltage_kv = _estimate_voltage_level(va, vb, vc) if va else None

        return DifferentialFeatures(
            di_dt_max=di_dt_max,
            di_dt_phase=di_dt_phase,
            peak_fault_current_a=peak_current,
            i0_i1_ratio=i0_i1_ratio,
            thd_percent=thd_percent,
            inception_angle_degrees=inception_angle,
            idiff_max_percent=idiff_max,
            irestraint_max_percent=irestraint_max,
            idiff_rise_rate=idiff_rise_rate,
            reclose_attempted=protection.auto_reclose_attempted,
            reclose_successful=protection.auto_reclose_successful,
            faulted_phases=fault.faulted_phases,
            fault_type=fault_type,
            is_ground_fault=is_ground_fault,
            station_name=record.station_name,
            relay_model=record.rec_dev_id,
            voltage_kv=voltage_kv,
            sampling_rate_hz=sampling_rate
        )

    except Exception as e:
        logger.error(f"Failed to extract differential features: {e}", exc_info=True)
        return None


# ===== HELPER FUNCTIONS =====

def _get_channel(record, name, measurement_type):
    """Get channel by canonical name and measurement type."""
    for ch in record.analog_channels:
        if ch.canonical_name == name:
            if measurement_type is None or ch.measurement == measurement_type:
                return ch
    return None


def _get_sampling_rate(record):
    """Calculate sampling rate from time array."""
    if len(record.time) > 1:
        dt = record.time[1] - record.time[0]
        if dt > 0:
            return 1.0 / dt
    return 0.0


def _calculate_impedance(va, vb, vc, ia, ib, ic, inception_idx, sampling_rate, system_freq, faulted_phases):
    """
    Calculate fault impedance Z = V/I using DFT at fundamental frequency.

    Returns:
        (r_x_ratio, z_magnitude, z_angle) or (None, None, None) if calculation fails
    """

    try:
        # Use 1 cycle after inception for impedance calculation
        samples_per_cycle = int(sampling_rate / system_freq)
        window_start = inception_idx
        window_end = min(inception_idx + samples_per_cycle, len(ia.samples))

        if window_end - window_start < samples_per_cycle // 2:
            logger.warning("Insufficient samples for impedance calculation")
            return None, None, None

        # Determine which phase to use (first faulted phase)
        if not faulted_phases:
            faulted_phases = ['A']  # Default

        phase = faulted_phases[0]

        if phase == 'A':
            v_samples = va.samples[window_start:window_end]
            i_samples = ia.samples[window_start:window_end]
        elif phase == 'B':
            v_samples = vb.samples[window_start:window_end]
            i_samples = ib.samples[window_start:window_end]
        else:  # C
            v_samples = vc.samples[window_start:window_end]
            i_samples = ic.samples[window_start:window_end]

        # Calculate phasors using DFT
        v_phasor = _calculate_phasor(v_samples, system_freq, sampling_rate)
        i_phasor = _calculate_phasor(i_samples, system_freq, sampling_rate)

        if i_phasor == 0 or np.abs(i_phasor) < 1.0:  # Avoid division by very small current
            logger.warning("Current too small for impedance calculation")
            return None, None, None

        # Z = V / I
        z_phasor = v_phasor / i_phasor

        # Convert to primary ohms (samples already in primary from parser)
        z_real = np.real(z_phasor) * 1000  # kV/A = milliohms, * 1000 = ohms
        z_imag = np.imag(z_phasor) * 1000

        z_magnitude = np.abs(z_phasor) * 1000
        z_angle = np.angle(z_phasor, deg=True)

        # R/X ratio
        if np.abs(z_imag) > 0.001:
            r_x_ratio = z_real / z_imag
        else:
            r_x_ratio = None

        return r_x_ratio, z_magnitude, z_angle

    except Exception as e:
        logger.error(f"Impedance calculation failed: {e}")
        return None, None, None


def _calculate_phasor(samples, system_freq, sampling_rate):
    """Calculate fundamental frequency phasor using DFT."""
    N = len(samples)
    if N == 0:
        return 0j

    # DFT at fundamental frequency
    n = np.arange(N)
    k_fundamental = system_freq * N / sampling_rate

    # DFT formula: X[k] = sum(x[n] * exp(-j*2*pi*k*n/N))
    phasor = np.sum(samples * np.exp(-2j * np.pi * k_fundamental * n / N)) / N

    # Multiply by 2 to get RMS amplitude (DFT gives peak/2)
    return phasor * 2


def _calculate_voltage_sag(va, vb, vc, inception_idx, sampling_rate, system_freq):
    """Calculate voltage sag depth and identify most affected phase."""

    try:
        # Pre-fault RMS (use 2 cycles before inception)
        samples_per_cycle = int(sampling_rate / system_freq)
        prefault_start = max(0, inception_idx - 2 * samples_per_cycle)
        prefault_end = inception_idx

        prefault_rms_a = np.sqrt(np.mean(va.samples[prefault_start:prefault_end]**2))
        prefault_rms_b = np.sqrt(np.mean(vb.samples[prefault_start:prefault_end]**2))
        prefault_rms_c = np.sqrt(np.mean(vc.samples[prefault_start:prefault_end]**2))

        # Fault RMS (use 2 cycles after inception)
        fault_start = inception_idx
        fault_end = min(inception_idx + 2 * samples_per_cycle, len(va.samples))

        fault_rms_a = np.sqrt(np.mean(va.samples[fault_start:fault_end]**2))
        fault_rms_b = np.sqrt(np.mean(vb.samples[fault_start:fault_end]**2))
        fault_rms_c = np.sqrt(np.mean(vc.samples[fault_start:fault_end]**2))

        # Calculate sag per phase
        sag_a = (prefault_rms_a - fault_rms_a) / prefault_rms_a if prefault_rms_a > 0 else 0
        sag_b = (prefault_rms_b - fault_rms_b) / prefault_rms_b if prefault_rms_b > 0 else 0
        sag_c = (prefault_rms_c - fault_rms_c) / prefault_rms_c if prefault_rms_c > 0 else 0

        # Find phase with deepest sag
        sags = {'A': sag_a, 'B': sag_b, 'C': sag_c}
        deepest_phase = max(sags, key=sags.get)
        deepest_sag = sags[deepest_phase]

        return deepest_sag, deepest_phase

    except Exception as e:
        logger.error(f"Voltage sag calculation failed: {e}")
        return 0.0, 'A'


def _calculate_di_dt(ia, ib, ic, inception_idx, sampling_rate, system_freq):
    """Calculate maximum dI/dt in first cycle after inception."""

    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        window_start = inception_idx
        window_end = min(inception_idx + samples_per_cycle, len(ia.samples))

        dt = 1.0 / sampling_rate

        # Calculate dI/dt for each phase
        di_dt_a = np.gradient(ia.samples[window_start:window_end], dt)
        di_dt_b = np.gradient(ib.samples[window_start:window_end], dt)
        di_dt_c = np.gradient(ic.samples[window_start:window_end], dt)

        # Find max absolute value
        max_di_dt_a = np.max(np.abs(di_dt_a))
        max_di_dt_b = np.max(np.abs(di_dt_b))
        max_di_dt_c = np.max(np.abs(di_dt_c))

        # Determine which phase has highest
        di_dts = {'A': max_di_dt_a, 'B': max_di_dt_b, 'C': max_di_dt_c}
        max_phase = max(di_dts, key=di_dts.get)
        max_value = di_dts[max_phase]

        return max_value, max_phase

    except Exception as e:
        logger.error(f"dI/dt calculation failed: {e}")
        return 0.0, 'A'


def _calculate_peak_current(ia, ib, ic, inception_idx, sampling_rate, system_freq):
    """Calculate peak fault current in first 2 cycles."""

    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        window_start = inception_idx
        window_end = min(inception_idx + 2 * samples_per_cycle, len(ia.samples))

        peak_a = np.max(np.abs(ia.samples[window_start:window_end]))
        peak_b = np.max(np.abs(ib.samples[window_start:window_end]))
        peak_c = np.max(np.abs(ic.samples[window_start:window_end]))

        peaks = {'A': peak_a, 'B': peak_b, 'C': peak_c}
        max_phase = max(peaks, key=peaks.get)
        max_value = peaks[max_phase]

        return max_value, max_phase

    except Exception as e:
        logger.error(f"Peak current calculation failed: {e}")
        return 0.0, 'A'


def _calculate_i0_i1_ratio(ia, ib, ic, inception_idx, sampling_rate, system_freq):
    """Calculate I0/I1 ratio using symmetrical components."""

    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        window_start = inception_idx
        window_end = min(inception_idx + samples_per_cycle, len(ia.samples))

        # Get phasors
        i_a = _calculate_phasor(ia.samples[window_start:window_end], system_freq, sampling_rate)
        i_b = _calculate_phasor(ib.samples[window_start:window_end], system_freq, sampling_rate)
        i_c = _calculate_phasor(ic.samples[window_start:window_end], system_freq, sampling_rate)

        # Symmetrical components
        # I0 = (Ia + Ib + Ic) / 3
        # I1 = (Ia + a*Ib + a^2*Ic) / 3, where a = exp(j*2*pi/3)
        a = np.exp(2j * np.pi / 3)

        i0 = (i_a + i_b + i_c) / 3
        i1 = (i_a + a * i_b + a**2 * i_c) / 3

        ratio = np.abs(i0) / np.abs(i1) if np.abs(i1) > 0.1 else 0.0

        return ratio

    except Exception as e:
        logger.error(f"I0/I1 calculation failed: {e}")
        return 0.0


def _calculate_thd(ia, ib, ic, inception_idx, sampling_rate, system_freq, faulted_phases):
    """Calculate THD of fault current in first 2 cycles."""

    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        window_start = inception_idx
        window_end = min(inception_idx + 2 * samples_per_cycle, len(ia.samples))

        # Use most faulted phase
        if not faulted_phases:
            faulted_phases = ['A']

        phase = faulted_phases[0]

        if phase == 'A':
            samples = ia.samples[window_start:window_end]
        elif phase == 'B':
            samples = ib.samples[window_start:window_end]
        else:
            samples = ic.samples[window_start:window_end]

        # FFT
        N = len(samples)
        fft_vals = fft(samples)
        freqs = fftfreq(N, 1/sampling_rate)

        # Find fundamental frequency bin
        fund_idx = np.argmin(np.abs(freqs - system_freq))
        fund_mag = np.abs(fft_vals[fund_idx])

        # Calculate harmonic content (excluding DC and fundamental)
        harmonic_sum_sq = np.sum(np.abs(fft_vals[2:N//2])**2) - fund_mag**2

        # Clamp to zero if negative (numerical errors)
        if harmonic_sum_sq < 0:
            harmonic_sum_sq = 0.0

        if fund_mag > 0:
            thd = np.sqrt(harmonic_sum_sq) / fund_mag * 100
        else:
            thd = 0.0

        return min(thd, 100.0)  # Cap at 100%

    except Exception as e:
        logger.error(f"THD calculation failed: {e}")
        return 0.0


def _calculate_inception_angle(va, vb, vc, inception_idx, sampling_rate, system_freq, faulted_phases):
    """Calculate voltage phase angle at fault inception."""

    try:
        # Use pre-fault voltage to establish reference
        samples_per_cycle = int(sampling_rate / system_freq)
        prefault_start = max(0, inception_idx - samples_per_cycle)
        prefault_end = inception_idx

        # Use faulted phase
        if not faulted_phases:
            faulted_phases = ['A']

        phase = faulted_phases[0]

        if phase == 'A':
            v_samples = va.samples[prefault_start:prefault_end]
            v_inception = va.samples[inception_idx]
        elif phase == 'B':
            v_samples = vb.samples[prefault_start:prefault_end]
            v_inception = vb.samples[inception_idx]
        else:
            v_samples = vc.samples[prefault_start:prefault_end]
            v_inception = vc.samples[inception_idx]

        # Get pre-fault phasor for reference
        v_phasor_prefault = _calculate_phasor(v_samples, system_freq, sampling_rate)
        v_magnitude = np.abs(v_phasor_prefault)

        if v_magnitude < 0.1:
            return 0.0

        # Estimate angle at inception based on instantaneous value
        # V(t) = Vmax * sin(wt + phi)
        # At inception: v_inception = Vmax * sin(phi)
        # phi = arcsin(v_inception / Vmax)
        v_peak = v_magnitude * np.sqrt(2)  # Convert RMS to peak
        sin_angle = v_inception / v_peak if v_peak > 0 else 0

        # Clamp to [-1, 1]
        sin_angle = np.clip(sin_angle, -1.0, 1.0)

        angle_rad = np.arcsin(sin_angle)
        angle_deg = np.degrees(angle_rad)

        # Convert to 0-360 range
        angle_deg = angle_deg % 360

        return angle_deg

    except Exception as e:
        logger.error(f"Inception angle calculation failed: {e}")
        return 0.0


def _determine_fault_type(faulted_phases, i0_i1_ratio):
    """Determine fault type from faulted phases and I0/I1 ratio."""

    n_phases = len(faulted_phases)
    is_ground = i0_i1_ratio > 0.3  # Ground fault if significant zero-sequence current

    if n_phases == 1:
        fault_type = "SLG" if is_ground else "LL"  # Single-phase could be SLG or LL
    elif n_phases == 2:
        fault_type = "DLG" if is_ground else "LL"
    elif n_phases == 3:
        fault_type = "3PH"
    else:
        fault_type = "UNKNOWN"

    return fault_type, is_ground


def _calculate_symmetrical_magnitudes(ia, ib, ic, inception_idx, sampling_rate, system_freq):
    """Returns (i0_mag, i1_mag, i2_mag) RMS magnitudes in primary amps."""
    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        ws = inception_idx
        we = min(inception_idx + samples_per_cycle, len(ia.samples))

        i_a = _calculate_phasor(ia.samples[ws:we], system_freq, sampling_rate)
        i_b = _calculate_phasor(ib.samples[ws:we], system_freq, sampling_rate)
        i_c = _calculate_phasor(ic.samples[ws:we], system_freq, sampling_rate)

        a = np.exp(2j * np.pi / 3)
        i0 = (i_a + i_b + i_c) / 3
        i1 = (i_a + a * i_b + a**2 * i_c) / 3
        i2 = (i_a + a**2 * i_b + a * i_c) / 3

        return float(np.abs(i0)), float(np.abs(i1)), float(np.abs(i2))
    except Exception as e:
        logger.error(f"Symmetrical magnitude calculation failed: {e}")
        return None, None, None


def _calculate_voltage_levels(va, vb, vc, inception_idx, sampling_rate, system_freq, faulted_phase='A'):
    """Returns (v_prefault_rms, v_fault_rms) in primary volts for the faulted phase."""
    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        prefault_start = max(0, inception_idx - 2 * samples_per_cycle)
        prefault_end = inception_idx
        fault_start = inception_idx
        fault_end = min(inception_idx + 2 * samples_per_cycle, len(va.samples))

        channels = {'A': va, 'B': vb, 'C': vc}
        v_ch = channels.get(faulted_phase, va)

        if prefault_end <= prefault_start or fault_end <= fault_start:
            return None, None

        v_pre = float(np.sqrt(np.mean(v_ch.samples[prefault_start:prefault_end]**2)))
        v_flt = float(np.sqrt(np.mean(v_ch.samples[fault_start:fault_end]**2)))
        return v_pre, v_flt
    except Exception as e:
        logger.error(f"Voltage level calculation failed: {e}")
        return None, None


def _estimate_voltage_level(va, vb, vc):
    """Estimate voltage level in kV from phase voltages."""

    try:
        # Use first 10% of recording as pre-fault
        n_samples = len(va.samples)
        prefault_samples = n_samples // 10

        rms_a = np.sqrt(np.mean(va.samples[:prefault_samples]**2))
        rms_b = np.sqrt(np.mean(vb.samples[:prefault_samples]**2))
        rms_c = np.sqrt(np.mean(vc.samples[:prefault_samples]**2))

        avg_rms = np.mean([rms_a, rms_b, rms_c])

        # Snap to nearest PLN nominal voltage level (primary kV)
        # If values are secondary (e.g. 57V from 150kV/√3 / 100V VT), scale up
        PLN_LEVELS = [30.0, 70.0, 150.0, 245.0, 500.0]
        if avg_rms > 350:
            return 500.0
        elif avg_rms > 180:
            return 245.0
        elif avg_rms > 100:
            return 150.0
        elif avg_rms > 45:
            return 70.0
        elif avg_rms > 20:
            return 30.0
        else:
            # Likely secondary voltage — cannot reliably determine nominal
            return None

    except Exception as e:
        logger.error(f"Voltage level estimation failed: {e}")
        return None
