# Legacy HPI engine (`legacy-mode`)

The `legacy-mode` branch adds an opt-in numerical HPI fitting engine intended for regression comparisons with the historical preprocessing pipeline. The existing/current engine remains the default; legacy behavior is selected explicitly with `--legacy`.

## Numerical behavior

- Adds `hpi/_legacy.py` with `fit_hpi_legacy(...)`, returning the shared fit-result fields consumed by the existing coregistration and diagnostic code, plus engine/inclusion metadata.
- Uses the configured fitting sample rate and a midpoint-centered crop spanning approximately two seconds around each coil activation.
- Uses the legacy integer-frequency peak-spacing behavior and duplicate-frequency amplitude model, then calls MNE's stock `compute_chpi_locs` localizer.
- Matches coil positions to digitised Polhemus positions using the legacy uncentred nearest-neighbour behavior and computes a closed-form rigid registration. It does not apply the current engine's field-GOF transform refinement.
- Uses a strict `GOF > 0.9` inclusion rule by default. An explicitly supplied `--gof` threshold uses an inclusive `GOF >= threshold` comparison.
- Reports `pol_gofs` as unavailable: the legacy engine does not calculate the current engine's fixed-position Polhemus GOF diagnostic.
- Accepts only verified head-frame FIF digitisation (or a canonical head-frame `DigMontage`). It rejects JSON and unverified dictionary digitisation inputs.

The legacy mode is limited to the numerical fitting/registration path. It does not recreate historical file discovery, logging, saving, or orchestration behavior.

## CLI behavior

- `hpi/coregister.py` and `hpi/check.py` accept `--legacy`; without it, the current engine remains in use.
- Legacy GOF default is `0.9`; the current engine's default remains `0.95`.
- Coregistered legacy outputs use a distinct `-legacy` suffix to avoid overwriting current-engine output names.
- Legacy mode does not support reference-file noise detection or the current engine's field-GOF refinement option. Coregistering with legacy mode requires head-frame FIF digitisation.

## Verification status

Synthetic regression tests cover crop/model behavior, inclusion thresholds, result schema, unsupported digitisation inputs, CLI selection, and recommendation behavior when Polhemus GOF is unavailable. These tests do not establish numerical parity on real recordings; real-data validation remains necessary.
