"""
Channel Name Normalization
===========================
Maps vendor-specific channel names to canonical names (VA, VB, VC, VN, IA, IB, IC, IN).
Handles SEL, ABB, Siemens, GE, and Qualitrol naming conventions.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Cache for loaded mappings
_CHANNEL_MAPPINGS: Optional[dict] = None


def load_channel_mappings(config_path: Optional[str] = None) -> dict:
    """Load channel mappings from JSON file (cached)."""
    global _CHANNEL_MAPPINGS

    if _CHANNEL_MAPPINGS is not None:
        return _CHANNEL_MAPPINGS

    if config_path is None:
        # Default to config/channel_mappings.json relative to this file
        config_path = Path(__file__).parent.parent / "config" / "channel_mappings.json"

    try:
        with open(config_path, 'r') as f:
            _CHANNEL_MAPPINGS = json.load(f)
        logger.info(f"Loaded channel mappings from {config_path}")
        return _CHANNEL_MAPPINGS
    except Exception as e:
        logger.error(f"Failed to load channel mappings: {e}")
        return {"manufacturers": {}, "manufacturer_detection": {}}


def detect_manufacturer(rec_dev_id: str, station_name: str = "") -> str:
    """
    Guess manufacturer from relay model ID or station name.

    Args:
        rec_dev_id: Relay device ID from .cfg file
        station_name: Station name from .cfg file

    Returns:
        Manufacturer name ("SEL", "ABB", "SIEMENS", "GE", "QUALITROL") or "UNKNOWN"
    """
    mappings = load_channel_mappings()
    detection_patterns = mappings.get("manufacturer_detection", {})

    # Combine device ID and station name for searching
    search_text = f"{rec_dev_id} {station_name}".upper()

    # ── Regex-based detection first (faster and more precise) ────────────────
    # Siemens SIPROTEC 5: device ID format BM + 10 digits (e.g. BM1906001619)
    # BM = "Bestellnummer Modell" — Siemens DIGSI 5 order number
    if re.match(r'^BM\d{8,12}$', rec_dev_id.strip()):
        logger.debug(f"Detected manufacturer: SIEMENS (BM order number: {rec_dev_id})")
        return "SIEMENS"

    for manufacturer, patterns in detection_patterns.items():
        for pattern in patterns:
            if pattern.upper() in search_text:
                logger.debug(f"Detected manufacturer: {manufacturer} (pattern: {pattern})")
                return manufacturer

    logger.warning(f"Could not detect manufacturer from '{rec_dev_id}' or '{station_name}'")
    return "UNKNOWN"


def normalize_channel_name(raw_name: str, unit: str, manufacturer: str = "UNKNOWN") -> Dict[str, Optional[str]]:
    """
    Normalize a vendor-specific channel name to canonical form.

    Args:
        raw_name: Original channel name from .cfg file
        unit: Channel unit ("kV", "V", "A", "kA", etc.)
        manufacturer: Detected manufacturer

    Returns:
        Dictionary with:
            - canonical_name: Standardized name (VA/VB/VC/VN/IA/IB/IC/IN)
            - phase: Phase identifier ("A", "B", "C", "N", or None)
            - measurement: "voltage" or "current"
    """
    # Normalize inputs
    raw_upper = raw_name.strip().upper()
    unit_upper = unit.strip().upper()

    # Determine measurement type from unit
    if any(v in unit_upper for v in ["KV", "V"]):
        measurement = "voltage"
    elif any(i in unit_upper for i in ["KA", "A"]):
        measurement = "current"
    else:
        measurement = "unknown"
        logger.warning(f"Unknown unit '{unit}' for channel '{raw_name}'")

    # Try manufacturer-specific patterns first
    mappings = load_channel_mappings()
    manufacturer_patterns = mappings.get("manufacturers", {}).get(manufacturer, {}).get("channel_patterns", {})

    for canonical, patterns in manufacturer_patterns.items():
        for pattern in patterns:
            if pattern.upper() == raw_upper or pattern.upper() in raw_upper:
                # Extract phase from canonical name
                phase = canonical[-1] if canonical[-1] in "ABCN" else None
                logger.debug(f"Matched '{raw_name}' → '{canonical}' (manufacturer: {manufacturer})")
                return {
                    "canonical_name": canonical,
                    "phase": phase,
                    "measurement": measurement
                }

    # Fall back to generic pattern matching
    canonical_name, phase = _generic_pattern_match(raw_upper, measurement)

    if canonical_name:
        logger.debug(f"Generic match '{raw_name}' → '{canonical_name}'")
        return {
            "canonical_name": canonical_name,
            "phase": phase,
            "measurement": measurement
        }

    # No match found - use raw name
    logger.warning(f"Could not normalize channel '{raw_name}' (unit: {unit}, mfr: {manufacturer})")
    return {
        "canonical_name": raw_name,
        "phase": None,
        "measurement": measurement
    }


def _generic_pattern_match(raw_upper: str, measurement: str) -> tuple:
    """
    Generic pattern matching for common channel naming conventions.
    Supports both ABC and RST phase naming.

    Returns:
        (canonical_name, phase) tuple
    """
    # RST phase notation (R=A, S=B, T=C) - check first to avoid confusion with VR (voltage regulator)
    # Need to be careful: VR could mean "phase R voltage" OR "voltage regulator"
    # Check word boundaries to distinguish
    words = raw_upper.split()
    if "VR" in words or raw_upper.startswith("VR ") or " VR" in raw_upper:
        return ("VA", "A")  # VR maps to VA (phase R = phase A)
    if "VS" in words or raw_upper.startswith("VS ") or " VS" in raw_upper:
        return ("VB", "B")  # VS maps to VB (phase S = phase B)
    if "VT" in words or raw_upper.startswith("VT ") or " VT" in raw_upper:
        return ("VC", "C")  # VT maps to VC (phase T = phase C)
    if "IR" in words or raw_upper.startswith("IR ") or " IR" in raw_upper:
        return ("IA", "A")  # IR maps to IA (phase R = phase A)
    if "IS" in words or raw_upper.startswith("IS ") or " IS" in raw_upper:
        return ("IB", "B")  # IS maps to IB (phase S = phase B)
    if "IT" in words or raw_upper.startswith("IT ") or " IT" in raw_upper:
        return ("IC", "C")  # IT maps to IC (phase T = phase C)

    # Neutral/residual patterns - check first to avoid false matches with phase patterns
    # (e.g., "IN" could be mistaken for "IC" if phase C is checked first)
    if measurement == "voltage" and any(p in raw_upper for p in ["VN", "V0", "3V0", "V_N", "V_RES", "RESVOL", "VG", "UE", "U0"]):
        return ("VN", "N")
    if measurement == "current" and any(p in raw_upper for p in ["IN", "I0", "3I0", "I_N", "I_RES", "RESCUR", "IG", "IE"]):
        return ("IN", "N")

    # Phase A patterns (ABC notation)
    if any(p in raw_upper for p in ["VA", "V_A", "V1", "VPHSA", "V PHASE A", "V A", "UL1", "UA"]):
        return ("VA", "A")
    if any(p in raw_upper for p in ["IA", "I_A", "I1", "IL1", "IPHSA", "I PHASE A", "I A"]):
        return ("IA", "A")

    # Phase B patterns (ABC notation)
    if any(p in raw_upper for p in ["VB", "V_B", "V2", "VPHSB", "V PHASE B", "V B", "UL2", "UB"]):
        return ("VB", "B")
    if any(p in raw_upper for p in ["IB", "I_B", "I2", "IL2", "IPHSB", "I PHASE B", "I B"]):
        return ("IB", "B")

    # Phase C patterns (ABC notation)
    if any(p in raw_upper for p in ["VC", "V_C", "V3", "VPHSC", "V PHASE C", "V C", "UL3", "UC"]):
        return ("VC", "C")
    if any(p in raw_upper for p in ["IC", "I_C", "I3", "IL3", "IPHSC", "I PHASE C", "I C"]):
        return ("IC", "C")

    return (None, None)
