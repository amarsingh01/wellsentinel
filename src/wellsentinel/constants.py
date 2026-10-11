"""Shared constants for the 3W dataset."""

from __future__ import annotations

PRESSURE_COLS: tuple[str, ...] = (
    "P-PDG",
    "P-TPT",
    "P-MON-CKP",
    "P-JUS-CKP",
    "P-MON-CKGL",
    "P-JUS-CKGL",
    "P-ANULAR",
    "P-JUS-BS",
    "P-MON-SDV-P",
    "PT-P",
)

TEMP_COLS: tuple[str, ...] = ("T-PDG", "T-TPT", "T-MON-CKP", "T-JUS-CKP")

FLOW_COLS: tuple[str, ...] = ("QGL", "QBS")

STATE_COLS: tuple[str, ...] = (
    "ESTADO-DHSV",
    "ESTADO-M1",
    "ESTADO-M2",
    "ESTADO-PXO",
    "ESTADO-SDV-GL",
    "ESTADO-SDV-P",
    "ESTADO-W1",
    "ESTADO-W2",
    "ESTADO-XO",
    "ABER-CKGL",
    "ABER-CKP",
)

ANALOG_COLS: tuple[str, ...] = PRESSURE_COLS + TEMP_COLS + FLOW_COLS

LABEL_COLS: tuple[str, ...] = ("class", "state")

#: 0 = normal; 1-9 = undesirable event types; 101-109 = transient onset of the same events.
NORMAL_CLASS = 0
TRANSIENT_OFFSET = 100

#: Human readable names, from dataset.ini.
CLASS_NAMES: dict[int, str] = {
    0: "NORMAL",
    1: "ABRUPT_INCREASE_OF_BSW",
    2: "SPURIOUS_CLOSURE_OF_DHSV",
    3: "SEVERE_SLUGGING",
    4: "FLOW_INSTABILITY",
    5: "RAPID_PRODUCTIVITY_LOSS",
    6: "QUICK_RESTRICTION_IN_PCK",
    7: "SCALING_IN_PCK",
    8: "HYDRATE_IN_PRODUCTION_LINE",
    9: "HYDRATE_IN_SERVICE_LINE",
}

#: Source tags inferred from the file name.
SOURCE_REAL = "real"
SOURCE_SIMULATED = "simulated"
SOURCE_DRAWN = "drawn"

#: Default interpolation limit. Gaps longer than this are flagged, not filled.
DEFAULT_MAX_GAP_SECONDS = 60
