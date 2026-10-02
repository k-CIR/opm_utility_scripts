"""Opt-in numerical HPI engine from ``legacy_opm_preprocess.py``.

This ports the amplitude/localisation part of ``find_hpi_fit`` and the
point-matching math of ``process_single_file``, not their file discovery,
logging, saving, or CLI orchestration. MNE's stock localiser and its full
duplicate-frequency GLM are deliberately retained.
"""

import os
import warnings

import mne
import numpy as np
from mne._fiff._digitization import _call_make_dig_points
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs
from mne.io.constants import FIFF
from mne.transforms import (
    Transform,
    _fit_matched_points,
    _quat_to_affine,
    apply_trans,
    get_ras_to_neuromag_trans,
)
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from ..channels import get_hpi_output_channels
from ..io import load_polhemus


def _load_head_polhemus(polfile):
    """Check frames before the shared loader discards that information."""
    if isinstance(polfile, mne.channels.DigMontage):
        dig = polfile.dig
        source = polfile
    elif isinstance(polfile, (str, os.PathLike)):
        source = os.fspath(polfile)
        if not source.lower().endswith('.fif'):
            raise ValueError('Legacy HPI requires head-frame FIF digitisation; '
                             'JSON and other formats are not supported.')
        dig = mne.io.read_info(source, verbose='error')['dig']
    else:
        raise TypeError('Legacy polfile must be a FIF path or head-frame '
                        'DigMontage, not a dict with unverified frames.')
    if not dig:
        raise ValueError('Legacy Polhemus input contains no digitisation points.')
    if any(d.get('coord_frame') != FIFF.FIFFV_COORD_HEAD for d in dig):
        raise ValueError('Legacy HPI only supports digitisation entirely in '
                         'the head coordinate frame.')
    if any(np.asarray(d['r']).shape != (3,) or
           not np.isfinite(d['r']).all() for d in dig):
        raise ValueError('Legacy digitisation points must be finite 3-vectors.')
    pol = load_polhemus(source)
    seed = get_ras_to_neuromag_trans(pol['nasion'], pol['lpa'], pol['rpa'])
    # The original FIF case assumes this conversion is identity. Do not
    # silently mix transformed localiser seeds with untransformed targets.
    if not np.allclose(seed, np.eye(4), atol=1e-6, rtol=0):
        raise ValueError('Legacy HPI requires canonical head-frame fiducials '
                         '(fiducial-to-head conversion must be identity).')
    return pol, seed


def _legacy_zero_location_channels(info):
    """Reproduce the original coordinate-sum test (including its flaw)."""
    picks = mne.pick_types(info, meg='mag', exclude='bads')
    return [info['chs'][p]['ch_name'] for p in picks
            if np.isclose(np.sum(info['chs'][p]['loc'][:3]), 0., atol=1e-3)]


def fit_hpi_legacy(hpifile, polfile, hpifreq, new_sfreq=1000, gof_limit=.9,
                   strict_legacy_gof=None):
    """Fit sequential HPI coils using the original numerical engine.

    Parameters
    ----------
    hpifile : path-like | mne.io.BaseRaw
        FIF HPI recording or Raw object (copied, never modified in place).
        Only device-frame magnetometers are supported.
    polfile : path-like | mne.channels.DigMontage
        FIF digitisation or montage entirely in canonical head coordinates.
        JSON and loader dicts are rejected because frame parity is unverified.
    hpifreq : float
        Shared drive frequency, including duplicates in MNE's amplitude GLM.
        Peak spacing uses ``round(sfreq / int(hpifreq)) - 2``, as in legacy.
    new_sfreq : float | None
        Fitting sample rate. None or zero disables resampling, as in legacy.
    gof_limit : float
        The default .9 uses strict ``gof > .9`` legacy inclusion. A different
        threshold uses ``gof >= gof_limit`` for configurable quality checks.

    Returns
    -------
    dict
        Common ``_core.fit_hpi`` apply/viz keys plus ``engine='legacy'``,
        ``legacy_inclusion`` (whether the stock strict .9 rule was used),
        ``hpi_freqs``, ``hpi_indices``, and the actual ``fit_sfreq``.
        ``pol_gofs`` contains NaNs: legacy has no fixed-position field-GOF
        calculation or field-GOF transform optimiser. Missing-peak coils
        retain their zero slope rows and ordering, as in the original.

    Notes
    -----
    Nearest-neighbour assignment is uncentred and not forced one-to-one.
    Failed amplitude fits and underdetermined rigid fits raise instead of
    being swallowed by the old orchestration's logging/continue blocks.
    """
    if isinstance(hpifreq, (bool, str)) or not np.isscalar(hpifreq):
        raise ValueError('hpifreq must be a finite scalar frequency >= 1 Hz.')
    hpifreq = float(hpifreq)
    if not np.isfinite(hpifreq) or hpifreq < 1:
        raise ValueError('hpifreq must be a finite scalar frequency >= 1 Hz.')
    if isinstance(gof_limit, (bool, str)) or not np.isscalar(gof_limit):
        raise ValueError('gof_limit must be a finite scalar between 0 and 1.')
    gof_limit = float(gof_limit)
    if not np.isfinite(gof_limit) or not 0 <= gof_limit <= 1:
        raise ValueError('gof_limit must be a finite scalar between 0 and 1.')
    if new_sfreq is not None:
        if isinstance(new_sfreq, (bool, str)) or not np.isscalar(new_sfreq):
            raise ValueError('new_sfreq must be None, zero, or a positive rate.')
        new_sfreq = float(new_sfreq)
        if not np.isfinite(new_sfreq) or new_sfreq < 0:
            raise ValueError('new_sfreq must be None, zero, or a positive rate.')

    pol, seed = _load_head_polhemus(polfile)
    if isinstance(hpifile, mne.io.BaseRaw):
        raw = hpifile.copy().load_data()
    elif isinstance(hpifile, (str, os.PathLike)):
        path = os.fspath(hpifile)
        if not path.lower().endswith(('.fif', '.fif.gz')):
            raise ValueError('Legacy hpifile must be a FIF recording.')
        raw = mne.io.read_raw_fif(path, preload=True, verbose='error')
    else:
        raise TypeError('Legacy hpifile must be a FIF path or MNE Raw object.')

    bads = list(dict.fromkeys(list(raw.info['bads']) +
                             _legacy_zero_location_channels(raw.info)))
    if bads:
        raw.drop_channels(bads)
    meg_picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    mag_picks = mne.pick_types(raw.info, meg='mag', exclude='bads')
    if not len(mag_picks) or not np.array_equal(meg_picks, mag_picks):
        raise ValueError('Legacy HPI requires MEG magnetometers only (no grads).')
    for p in mag_picks:
        ch = raw.info['chs'][p]
        if ch['coord_frame'] != FIFF.FIFFV_COORD_DEVICE:
            raise ValueError('Legacy HPI requires device-frame MEG sensors.')
        if not np.isfinite(ch['loc']).all():
            raise ValueError('Legacy HPI requires finite MEG sensor geometry.')
    if not len(mne.pick_types(raw.info, misc=True, exclude='bads')):
        raise ValueError('Legacy HPI requires MISC HPI output channels.')
    hpi_names, hpi_indices = get_hpi_output_channels(raw)
    n_hpi = len(hpi_indices)
    if n_hpi < 3:
        raise ValueError('Legacy HPI requires at least 3 active output channels.')
    if len(pol['hpi_orig']) < n_hpi:
        raise ValueError(f'Polhemus has {len(pol["hpi_orig"])} HPI points but '
                         f'{n_hpi} active HPI channels were detected.')

    if new_sfreq:
        raw.resample(new_sfreq, verbose='error')
    peak_dist = round(raw.info['sfreq'] / int(hpifreq)) - 2
    if peak_dist < 1:
        raise ValueError('Legacy integer-frequency peak spacing is < 1 sample; '
                         'increase new_sfreq or reduce hpifreq.')
    if hpifreq > min(raw.info['sfreq'] / 2, raw.info['lowpass']):
        raise ValueError('HPI frequency exceeds the fitting lowpass/Nyquist.')
    raw.info.update(dev_head_t=Transform('meg', 'head', seed))
    # Legacy used *all* Polhemus dig points as extra points for its seed,
    # including fiducials/HPI/EEG. Preserve this here, but return the shared
    # loader's classified extra/eeg points for apply-compatible digitisation.
    all_dig = np.array([d['r'] for d in pol['dig']], dtype=float)
    with raw.info._unlock():
        raw.info['dig'], _ = _call_make_dig_points(
            pol['nasion'], pol['lpa'], pol['rpa'],
            pol['hpi_orig'][:n_hpi], all_dig, convert=True,
        )
        raw.info['hpi_subsystem'] = {
            'hpi_coils': [{'event_bits': [256]} for _ in range(n_hpi)]}
        raw.info['hpi_meas'] = [{'hpi_coils': [
            dict(number=i + 1, drive_chan=name, coil_freq=hpifreq)
            for i, name in enumerate(hpi_names)]}]
        raw.info['hpi_results'] = [dict(
            dig_points=[dict(r=np.zeros(3), coord_frame=FIFF.FIFFV_COORD_DEVICE,
                             ident=i + 1) for i in range(n_hpi)],
            coord_trans=Transform('meg', 'head'),
        )]
        raw.info['line_freq'] = None

    raw_orig = raw.copy()
    slope = np.zeros((n_hpi, len(mag_picks)), dtype=float)
    coil_amplitudes = None
    peak_tmax = []
    for index, channel_index in enumerate(hpi_indices):
        raw = raw_orig.copy()
        b = raw[channel_index, :][0].ravel()
        peaks, _ = find_peaks(b, distance=peak_dist, height=.0001)
        if not peaks.size:
            warnings.warn(f'No peaks found for {hpi_names[index]}; retaining '
                          'legacy zero slope row.', RuntimeWarning, stacklevel=2)
            continue
        min_t, max_t = peaks[[0, -1]] / raw.info['sfreq']
        midpoint = (max_t - min_t) / 2 + min_t
        if midpoint - 1 < 0 or midpoint + 1 > raw.times[-1]:
            raise ValueError(f'Legacy midpoint +/- 1 s crop for '
                             f'{hpi_names[index]} exceeds the recording bounds.')
        peak_tmax.append(max_t)
        raw.crop(tmin=midpoint - 1, tmax=midpoint + 1, verbose='error')
        coil_amplitudes = compute_chpi_amplitudes(
            raw, tmin=0, tmax=2, t_window=2, t_step_min=2, verbose='error')
        if np.shape(coil_amplitudes['slopes']) != (1, n_hpi, len(mag_picks)):
            raise RuntimeError('Stock MNE returned an unsupported legacy '
                               'amplitude shape (expected one full-coil window).')
        slope[index] = coil_amplitudes['slopes'][0][index]
    if coil_amplitudes is None:
        raise ValueError('No HPI output channel had peaks for legacy fitting.')
    coil_amplitudes['slopes'][0] = slope
    coil_locs = compute_chpi_locs(raw.info, coil_amplitudes, verbose='error')
    if len(coil_locs['times']) != 1:
        raise RuntimeError('Stock MNE returned no single legacy localisation.')
    hpi_dev = np.asarray(coil_locs['rrs'][0])
    hpi_gofs = np.asarray(coil_locs['gofs'][0])
    if not np.isfinite(hpi_dev).all() or not np.isfinite(hpi_gofs).all():
        raise ValueError('Legacy localisation returned non-finite positions/GOFs.')

    if strict_legacy_gof is None:
        strict_legacy_gof = gof_limit == .9
    legacy_inclusion = bool(strict_legacy_gof)
    include_hpis = hpi_gofs > .9 if legacy_inclusion else hpi_gofs >= gof_limit
    if include_hpis.sum() < 3:
        raise ValueError('At least 3 HPI coils must pass the legacy GOF '
                         f'threshold ({gof_limit:g}) for a rigid transform.')
    _, tree_indices = cKDTree(pol['hpi_orig']).query(hpi_dev[include_hpis])
    dev_pts = hpi_dev[include_hpis]
    pol_pts = pol['hpi_orig'][tree_indices]
    if any(np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2
           for pts in (dev_pts, pol_pts)):
        raise ValueError('Legacy KD assignment gives an underdetermined rigid '
                         'fit (fewer than 3 non-collinear matched points).')
    trans = _quat_to_affine(_fit_matched_points(dev_pts, pol_pts)[0])
    dev_to_head_trans = Transform('meg', 'head', trans)
    dist = np.linalg.norm(pol_pts - apply_trans(dev_to_head_trans, dev_pts), axis=1)
    raw_for_topomap = raw_orig.copy().pick(picks=['meg'], exclude='bads')
    return {
        'dev_to_head_trans': dev_to_head_trans,
        'hpi_dev': hpi_dev, 'hpi_gofs': hpi_gofs,
        'hpi_orig': pol['hpi_orig'], 'hpi_names': hpi_names,
        'nasion': pol['nasion'], 'lpa': pol['lpa'], 'rpa': pol['rpa'],
        'pol_info': pol, 'extra_pts': pol['extra_pts'], 'eeg_pts': pol['eeg_pts'],
        'slope': slope, 'raw_for_topomap': raw_for_topomap,
        'dist': dist, 'include_hpis': include_hpis, 'tree_indices': tree_indices,
        'pol_gofs': np.full(include_hpis.sum(), np.nan),
        'bads': bads, 'bads_fig': None, 'optim': 'none',
        'engine': 'legacy', 'legacy_inclusion': legacy_inclusion,
        'gof_limit': gof_limit,
        'gof_inclusive': not legacy_inclusion,
        'hpi_freqs': np.full(n_hpi, hpifreq), 'hpi_indices': hpi_indices,
        'fit_sfreq': raw_orig.info['sfreq'], 'peak_tlast': max(peak_tmax),
    }
