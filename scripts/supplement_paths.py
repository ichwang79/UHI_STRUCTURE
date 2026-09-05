"""Path resolution for the supplementary-analysis scripts. Set the environment variables:
  UHI_AIR_DATA        the air-temperature UHI record (Zenodo 10.5281/zenodo.22006932)
  UHI_AIR_COMPANION   the companion city panels (Zenodo 10.5281/zenodo.22108287, v2 or later)
  UHI_CODE_INPUTS     data/inputs of the UHI_STRUCTURE code deposit after make_inputs.py has run
  UHI_CODE_SCRIPTS    scripts/ of the UHI_STRUCTURE code deposit (for gdp_rcs.py)
The extra inputs (GEE extractions, MTUC subset, crosswalks, seasonal panels) ship with version 3 of
the companion record; UHI_EXTRA_INPUTS overrides where they are read from and defaults to UHI_AIR_COMPANION."""
import os
_here = os.path.dirname(os.path.abspath(__file__))
def _env(k, default=None):
    v = os.environ.get(k, default)
    if v is None: raise SystemExit(f"set {k} (see supplement_paths.py)")
    return v.rstrip("/") + "/"
AIR = _env("UHI_AIR_DATA"); COMP = _env("UHI_AIR_COMPANION"); CODE_IN = _env("UHI_CODE_INPUTS", os.path.join(_here, "..", "data", "inputs"))
CODE_SCRIPTS = _env("UHI_CODE_SCRIPTS", _here)
EXTRA = _env("UHI_EXTRA_INPUTS", COMP)   # the version-2 companion record ships the extra inputs
