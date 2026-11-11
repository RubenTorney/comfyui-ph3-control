"""Human-readable strength presets, calibrated around the measured default (strength 1.2, steps 0 to 50 %).

All three presets were measured (6 poses x 2 seeds, character reference in Picture 1; PCK@0.1, sharpness relative to
the control-free reference): Light 1.2 / 0-25 % -> 0.792, 1.06 ; Normal 1.2 / 0-50 % -> 0.815, 1.15 ;
Strict 1.2 / 0-75 % -> 0.815, 1.06. Higher strengths were measured WORSE (1.5 / 0-37.5 % -> 0.726), so there is no
"very strict" preset: raise the end, not the strength.
"""
CUSTOM = "Custom (advanced)"
# name -> (strength, end as a fraction of the steps)
PRESETS = {
    "Light (Légère) 1.2 · 25 %": (1.2, 0.25),
    "Normal (Normale) 1.2 · 50 %": (1.2, 0.5),
    "Strict (Stricte) 1.2 · 75 %": (1.2, 0.75),
}
DEFAULT_PRESET = "Normal (Normale) 1.2 · 50 %"
CHOICES = list(PRESETS) + [CUSTOM]
DEFAULT_STRENGTH, DEFAULT_END = PRESETS[DEFAULT_PRESET]


def resolve(preset: str, strength: float, start_percent: float, end_percent: float) -> tuple[float, float, float]:
    """(strength, start, end), start / end as a fraction of the steps. A preset ignores the advanced fields."""
    if preset == CUSTOM:
        s, e = float(start_percent), float(end_percent)
        if e < s:
            raise ValueError(f"end_percent ({e}) must not be below start_percent ({s})")
        return float(strength), s, e
    if preset not in PRESETS:
        raise ValueError(f"unknown intensity: {preset!r}")
    st, end = PRESETS[preset]
    return st, 0.0, end
