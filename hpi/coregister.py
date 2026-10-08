"""
Unified HPI coregistration entry point.

Handles both single-file and multi-file cases through one script.
Any argument not supplied on the command line falls back to an interactive
CLI prompt, so the script works fully non-interactively, fully
interactively, or anywhere in between.

Usage (fully non-interactive)::

    python -m opm_utility_scripts.hpi.coregister \\
        --data  /path/to/AudOdd_raw.fif /path/to/RSEO_raw.fif \\
        --hpi   /path/to/HPIBefore_raw.fif \\
        --pol   /path/to/digitisation_sub-001_20260811140000.json \\
        --freq  33 \\
        --sfreq 1000 \\
        --save --overwrite --plot

Usage (fully interactive — CLI prompts for everything)::

    python -m opm_utility_scripts.hpi.coregister
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import mne
import numpy as np
from ..viz import plot_hpi_alignment
from ._core import fit_hpi, apply_transform, save_raw, write_settings_json
from ._options import add_fit_options, fit_options


# ---------------------------------------------------------------------------
# CLI input helpers (replace the old tkinter dialogs)
# ---------------------------------------------------------------------------

def _strip_quotes(raw):
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        return raw[1:-1]
    return raw


def _prompt_path(title, extensions=(), allow_blank=False):
    """Prompt for a single existing file path. Re-prompts until valid.

    When ``allow_blank`` is ``True``, an empty response returns ``None``
    instead of re-prompting (used for genuinely optional inputs).
    """
    hint_parts = []
    if extensions:
        hint_parts.append(', '.join(extensions))
    if allow_blank:
        hint_parts.append('leave blank to skip')
    hint = f' ({"; ".join(hint_parts)})' if hint_parts else ''
    while True:
        try:
            raw = input(f'{title}{hint}: ').strip()
        except EOFError:
            return None
        if not raw:
            if allow_blank:
                return None
            print('  Please enter a path.')
            continue
        path = os.path.expanduser(_strip_quotes(raw))
        if not os.path.isfile(path):
            print(f'  File not found: {path}')
            continue
        return path


def _prompt_paths(title, extensions=()):
    """Prompt for one or more existing file paths (comma-separated).

    Returns a tuple of paths, matching the shape returned by the old
    ``filedialog.askopenfilenames`` GUI helper.
    """
    hint = f' ({", ".join(extensions)})' if extensions else ''
    while True:
        try:
            raw = input(f'{title}{hint} [comma-separated if more than one]: ').strip()
        except EOFError:
            return ()
        if not raw:
            print('  Please enter at least one path.')
            continue
        candidates = [os.path.expanduser(_strip_quotes(p.strip()))
                      for p in raw.split(',') if p.strip()]
        missing = [p for p in candidates if not os.path.isfile(p)]
        if missing:
            for p in missing:
                print(f'  File not found: {p}')
            continue
        return tuple(candidates)


def _prompt_value(prompt, default):
    """Prompt for a free-text value, returning ``default`` when left blank."""
    try:
        raw = input(f'{prompt} [{default}]: ').strip()
    except EOFError:
        return default
    return raw or default


def _prompt_yes_no(prompt, default=False):
    """Prompt for a yes/no answer, returning ``default`` on blank input."""
    default_str = 'y' if default else 'n'
    while True:
        try:
            raw = input(f'{prompt} (y/n) [{default_str}]: ').strip().lower()
        except EOFError:
            return default
        if not raw:
            return default
        if raw in ('y', 'n'):
            return raw == 'y'
        print("  Please enter 'y' or 'n'.")


def _output_suffix(datfile: str, new_sfreq: float) -> str:
    """Return the output suffix, including '+ds' only when the file is resampled."""
    info = mne.io.read_info(datfile, verbose='error')
    suffix = '_proc-hpi'
    if int(new_sfreq) != int(info['sfreq']):
        suffix += '+ds'
    return suffix + '_raw.fif'


def _settings_sidecar_path(output_file: str, settings_json=None) -> str:
    """Name a sidecar from its transformed data output path.

    By default, ``name_proc-hpi_raw.fif`` maps to
    ``hpi_name_proc-hpi_raw.json`` in the same directory. If a sidecar
    directory is supplied, keep the same basename there.
    """
    directory = settings_json or os.path.dirname(output_file)
    basename = 'hpi_' + os.path.splitext(os.path.basename(output_file))[0] + '.json'
    return os.path.join(directory, basename)


def _parse_args():
    p = argparse.ArgumentParser(
        prog='python -m opm_utility_scripts.hpi.coregister',
        description=(
            'HPI coregistration.  Any omitted argument prompts interactively '
            'on the command line.'
        ),
    )
    p.add_argument(
        '--data', '-d', nargs='+', metavar='FILE',
        help='One or more data files to apply the transform to.',
    )
    p.add_argument(
        '--hpi', '-H', metavar='FILE',
        help='Raw HPI recording (e.g. HPIBefore_raw.fif).',
    )
    p.add_argument(
        '--pol', '-p', metavar='FILE',
        help='Polhemus digitisation file (.json or .fif).',
    )
    p.add_argument(
        '--reffile', '-r', metavar='FILE', default=None,
        help='Optional reference recording (e.g. resting state) used for '
             'background-power-based noisy channel detection '
             '(default: use a clean HPI tail when no reference is supplied).',
    )
    p.add_argument(
        '--freq', '-f', type=float, default=None, metavar='HZ',
        help='HPI drive frequency in Hz (default: ask).',
    )
    p.add_argument(
        '--gof', type=float, default=0.95, metavar='THRESH',
        help='Minimum dipole GOF for a coil to be included in the '
             'device-to-head transform fit (default: 0.95).',
    )
    add_fit_options(p)
    # Coregister always records fit provenance beside each saved output.
    # An explicit --settings-json DIR only changes the sidecar directory.
    p.set_defaults(settings_json='')
    p.add_argument(
        '--sfreq', '-s', type=float, default=None, metavar='HZ',
        help='Target sampling frequency in Hz (default: ask).',
    )
    p.add_argument(
        '--save', action='store_true', default=False,
        help='Save the transformed file(s) to disk (default: do not save).',
    )
    p.add_argument(
        '--overwrite', action='store_true', default=False,
        help='Overwrite existing output files (default: skip).',
    )
    p.add_argument(
        '--plot', action='store_true', default=False,
        help='Show and save the HPI alignment plot (default: no plot).',
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # ----------------------------------------------------------------
    # Resolve inputs — CLI args take priority; fall back to interactive
    # command-line prompts for anything not supplied.
    # ----------------------------------------------------------------
    need_prompts = not all([args.data, args.hpi, args.pol,
                            args.freq is not None, args.sfreq is not None])

    datafiles = args.data or _prompt_paths("Select data file(s)", extensions=('.fif',))
    hpifile   = args.hpi  or _prompt_path("Select HPI file", extensions=('.fif',))
    polfile   = args.pol  or _prompt_path("Select Polhemus file", extensions=('.json', '.fif'))

    # Reference file is genuinely optional — only asked interactively (with
    # a blank-to-skip option), never forced.
    reffile = args.reffile
    if reffile is None and need_prompts:
        reffile = _prompt_path(
            "Reference file for noisy-channel detection",
            extensions=('.fif',), allow_blank=True,
        )

    hpifreq   = args.freq  if args.freq  is not None \
                else float(_prompt_value("HPI frequency (Hz)", "33"))
    new_sfreq = args.sfreq if args.sfreq is not None \
                else float(_prompt_value("Downsampling frequency (Hz)", "1000"))

    # Save / overwrite / plot: if any of the three flags were given on the CLI,
    # use all CLI values directly.  Only ask interactively when prompts were
    # needed above (i.e. at least one file/freq arg was missing) and no flags
    # were provided.
    any_flags = args.save or args.overwrite or args.plot
    if any_flags:
        doSave     = args.save
        overwrite  = args.overwrite
        plotResult = args.plot
    elif need_prompts:
        doSave     = _prompt_yes_no("Save result to disk?", default=False)
        overwrite  = _prompt_yes_no("Overwrite existing files?", default=False) if doSave else False
        plotResult = _prompt_yes_no("Plot alignment?", default=False)
    else:
        doSave = overwrite = plotResult = False

    if not datafiles or not hpifile or not polfile:
        print("ERROR: data file(s), HPI file, and Polhemus file are all required.")
        sys.exit(1)

    print(f"Data file(s): {datafiles}")
    print(f"HPI file:     {hpifile}")
    print(f"Polhemus:     {polfile}")
    print(f"Reference:    {reffile if reffile else '(none supplied)'}")
    print(f"Frequency:    {hpifreq} Hz")
    print(f"GOF limit:    {args.gof}")
    selected_options = fit_options(args)
    settings_json = selected_options.pop('settings_json')
    print(f"Policies:     {selected_options}")
    print(f"Settings JSON: {'disabled' if settings_json is None else ('beside each output' if settings_json == '' else settings_json)}")
    print(f"Optimization: {args.optimization}")
    print(f"Target sfreq: {new_sfreq} Hz")
    print(f"Save:         {doSave}{'  (overwrite)' if overwrite else ''}")
    print(f"Plot:         {plotResult}")

    # ----------------------------------------------------------------
    # Fit HPI coils (shared across all data files)
    # ----------------------------------------------------------------
    fit = fit_hpi(hpifile, polfile, hpifreq, gof_limit=args.gof, reffile=reffile,
                  **selected_options)

    hpi_names       = fit['hpi_names']
    hpi_dev         = fit['hpi_dev']
    hpi_gofs        = fit['hpi_gofs']
    hpi_orig        = fit['hpi_orig']
    dist            = fit['dist']
    slope           = fit['slope']
    raw_for_topomap = fit['raw_for_topomap']

    # ----------------------------------------------------------------
    # Optional topomap (only shown when multiple files were selected)
    # ----------------------------------------------------------------
    if len(datafiles) > 1:
        fig_topo  = plt.figure(figsize=(13, 7))
        n_topomap = min(len(hpi_names), 4)
        for i in range(n_topomap):
            tmp = np.reshape(slope[i], (slope[i].size, 1))
            evo = mne.EvokedArray(tmp, raw_for_topomap.info)
            ax  = fig_topo.add_subplot(1, n_topomap, i + 1)
            evo.plot_topomap(0.0, ch_type='mag', size=3, res=512,
                             axes=ax, colorbar=False, show=False)
            ax.set_title(hpi_names[i], fontsize=14)
        plt.show()

    # ----------------------------------------------------------------
    # Print fit quality
    # ----------------------------------------------------------------
    print('---------------------------------------------')
    print(f"hpi_orig (head frame, mm):\n{np.round(hpi_orig * 1000, 1)}\n")
    print(f"hpi_dev  (device frame, mm):\n{np.round(hpi_dev * 1000, 1)}\n")
    print(f"mean distance = {np.mean(dist) * 1000:.1f} mm\n")
    for index, value in enumerate(hpi_gofs):
        status = 'ok' if fit['include_hpis'][index] else 'not ok'
        print(f"Coil: {hpi_names[index][-3:]}, GOF: {value:.3f}, Status: {status}")
    print('---------------------------------------------')

    # ----------------------------------------------------------------
    # Apply transform; optionally save
    # ----------------------------------------------------------------
    last_outpath = None
    last_datfile = None
    last_raw_out = None

    for datfile in datafiles:
        suffix  = _output_suffix(datfile, new_sfreq)
        raw_out = apply_transform(datfile, fit, new_sfreq)
        stem = os.path.splitext(os.path.basename(datfile))[0].replace('_raw', '')
        expected_outpath = os.path.join(os.path.dirname(datfile), stem + suffix)
        sidecar_allowed = True

        if doSave:
            outpath = expected_outpath
            if not overwrite and os.path.exists(outpath):
                print(f"Skipped (already exists): {outpath}")
                sidecar_allowed = False
            else:
                outpath = save_raw(raw_out, datfile, suffix, overwrite=overwrite)
                print(f"Saved: {outpath}")
                last_outpath = outpath

        if doSave and settings_json is not None and sidecar_allowed:
            sidecar_path = _settings_sidecar_path(expected_outpath, settings_json)
            write_settings_json(sidecar_path, fit, hpifile=hpifile, polfile=polfile,
                                reffile=reffile, datafile=datfile,
                                output_file=expected_outpath)
            print(f"Settings:     {sidecar_path}")

        last_datfile = datfile
        last_raw_out = raw_out

    # ----------------------------------------------------------------
    # Optional alignment plot
    # ----------------------------------------------------------------
    if plotResult and last_raw_out is not None:
        raw_hpi_for_plot = mne.io.read_raw_fif(hpifile, preload=False, verbose='error')

        ref_path  = last_outpath if last_outpath is not None else last_datfile
        plot_stem = os.path.splitext(ref_path)[0] if ref_path else 'hpi_alignment'
        plot_path = f"{plot_stem}_hpi_alignment.png"

        fig = plot_hpi_alignment(fit, raw=raw_hpi_for_plot, show=True, gof_threshold=args.gof)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        #print(f"Alignment plot saved: {plot_path}")


if __name__ == '__main__':
    main()
