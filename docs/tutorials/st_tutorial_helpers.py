"""Helpers for the ddm_st / ddm_normalt tutorials.

Extracted verbatim from the LAN pipeline's hssm_run_defaults.py (the canonical settings
file) so the tutorials are self-contained. Two public functions:

    make_manual_prior(param, linked, mu_sigma=None)  explicit link-space prior for an include term
    make_initvals(model, t_natural, bounds)          initial values that respect the link and bounds
"""
from __future__ import annotations
import numpy as np
from bambi.priors import Prior

HDDM_MU: dict = {
    "v": {"dist": "Normal", "mu": 2.0, "sigma": 3.0},
    "a": {"dist": "Gamma", "mu": 1.5, "sigma": 0.75},
    "z": {"dist": "Beta", "alpha": 10, "beta": 10},
    "t": {"dist": "Gamma", "mu": 0.2, "sigma": 0.2},
    "st": {"dist": "HalfNormal", "sigma": 0.3},
}

HDDM_SETTINGS_GROUP: dict = {
    "v": {"dist": "Normal", "mu": HDDM_MU["v"], "sigma": {"dist": "HalfNormal", "sigma": 2.0}},
    "a": {"dist": "Gamma", "mu": HDDM_MU["a"], "sigma": {"dist": "HalfNormal", "sigma": 0.1}},
    "z": {"dist": "Beta", "alpha": {"dist": "Gamma", "mu": 10, "sigma": 10},
          "beta": {"dist": "Gamma", "mu": 10, "sigma": 10}},
    "t": {"dist": "Gamma", "mu": HDDM_MU["t"], "sigma": {"dist": "HalfNormal", "sigma": 0.2}},
}

GENERIC_LINKSPACE_GROUP_PRIOR: dict = {
    "dist": "Normal",
    "mu": {"dist": "Normal", "mu": 0.0, "sigma": 0.25},
    "sigma": {"dist": "Weibull", "alpha": 1.5, "beta": 0.3},
}

T_INIT_NATURAL: float = 0.30

def make_initvals(model, t_natural: float = T_INIT_NATURAL, bounds=None, verbose: bool = True) -> dict:
    """HSSM's initvals with the t location set to `t_natural`, link-aware.

    Handles all four cases that occur in this codebase:
      * `1 + (1|subject)`   -> key is `t_Intercept`
      * `0 + (1|subject)`   -> key is `t_1|subject_mu`
      * identity link       -> value passed through unchanged
      * log_logit link      -> value transformed into link space (see note 4)
    """
    iv = dict(model.initvals)

    linked = False
    try:  # HSSM stores the link per parameter; identity has no `.link` bounds
        lk = model.params["t"].link
        linked = lk is not None and getattr(lk, "name", "") not in ("identity", "")
    except Exception:
        linked = False

    value = float(t_natural)
    if linked:
        from hssm.link import Link
        if bounds is None:
            bounds = tuple(model.params["t"].bounds)
        value = float(Link("gen_logit", bounds=tuple(bounds)).link(t_natural))

    for key in ("t_Intercept", "t_1|subject_mu", "t"):
        if key in iv:
            iv[key] = np.array(value, dtype=np.asarray(iv[key]).dtype)
            if verbose:
                space = f"link space (== {t_natural} natural)" if linked else "natural"
                print(f"  [hssm_run_defaults] initval {key} = {value:.4f} {space}")
            return iv

    raise KeyError(
        f"no t location initval found; available keys = {sorted(iv)}. "
        "Set it explicitly rather than letting HSSM's out-of-bounds default stand."
    )

def _build_prior(spec, bounds=None, _top=True):
    """Recursively turn a plain dict spec (`{"dist": ..., <kwarg>: <spec-or-number>}`)
    into a real `bambi.priors.Prior` object. Internal helper for `make_manual_prior()`.

    `bounds` is only ever attached at the TOP level, and only if not None -
    NOT to every nested hyperprior. Found 2026-08-25 (SMC diagnostic script
    construction): some PyMC RV ops (e.g. `Weibull`'s, class name
    `WeibullBetaRV`) raise `TypeError: unexpected keyword argument 'bounds'`
    if a `bounds` kwarg reaches them at all, even `bounds=None` - unlike
    e.g. `HalfNormal`/`Gamma`, which silently tolerate it. Passing
    `bounds=None` unconditionally at every recursion level (the original,
    buggy version of this function) happened to work for every Gamma/Beta/
    HalfNormal-shaped manual prior tried so far (job 5191214/5197145) but
    broke immediately on the generic Normal/Weibull link-space shape - i.e.
    it was silently correct by luck of which distributions got tested
    first, not by design.
    """
    if isinstance(spec, (int, float)):
        return spec
    spec = dict(spec)
    dist_name = spec.pop("dist")
    kwargs = {k: _build_prior(v, _top=False) for k, v in spec.items()}
    if _top and bounds is not None:
        kwargs["bounds"] = bounds
    return Prior(dist_name, **kwargs)

def make_manual_prior(param: str, linked: bool, mu_sigma: float | None = None, verbose: bool = True) -> Prior:
    """Hand-built, literal group-level ('1|subject', centered) prior for one
    HIER param - built from this file's OWN frozen constants (HDDM_MU /
    HDDM_SETTINGS_GROUP / GENERIC_LINKSPACE_GROUP_PRIOR above), never by
    calling `hssm.prior.get_hddm_default_prior`/`generate_prior`. Added
    2026-08-24 per Eitan's explicit instruction: stop relying on
    `prior_settings="safe"`'s auto-fill at all - it only ever fires when a
    term has no manually-supplied prior, so supplying one via this
    function for EVERY `include=[...]` entry makes that branch
    unreachable, categorically, rather than something to keep auditing.

    `linked`: MUST match whatever `link_settings` you're passing to
    `hssm.HSSM(...)` for this construction.
      - `linked=True`: returns the GENERIC_LINKSPACE_GROUP_PRIOR shape
        (Normal(mu=Normal(0,0.25), sigma=Weibull(1.5,0.3))) for every
        param, regardless of which param it is. Do NOT substitute a
        Gamma/Beta HDDM-shaped prior here - built and tested exactly that
        combination 2026-08-24 (`archive/26_ddm_normalt_hssm_wiring/
        prior_matrix_normalt_forcedprior.py`, job 5191214) and got an
        immediate frozen-chain collapse (step size 1.18e-38 from
        iteration 1, tree_depth=1, 100% divergence) at every st tested -
        working explanation is that a Gamma/Beta prior's own positive/
        unit-interval support constrains the SAME raw variable that also
        gets `gen_logit`'s sigmoid applied downstream, which can only ever
        land in the upper half of the sigmoid's range - a double-
        constraint composition, not a free choice of prior shape. See
        project_hssm_run_defaults memory for the full writeup and the
        one open caveat (initval confound not yet fully ruled out).
      - `linked=False`: returns the true HDDM-shaped prior for `param`
        from `HDDM_SETTINGS_GROUP` (Gamma for `a`/`t`, Beta for `z`,
        Normal for `v`) - only valid/coherent when nothing downstream
        re-transforms the same raw variable.

    `mu_sigma`: only used when `linked=True`. Overrides the mu-hyperprior's
    own sigma (default 0.25, i.e. GENERIC_LINKSPACE_GROUP_PRIOR unchanged).
    Added 2026-08-25 - this is the exact widened-prior lever
    (`single_subject_widened_check.py`/`mode_D_widened.py`, 2026-08-20/21:
    sigma 0.25->2.0 in link space) that previously showed real improvement
    for v/a/z recovery at N=1 under NUTS, re-exposed here through the
    canonical function instead of a one-off hand-rolled dict. A narrow
    prior can dominate a weak likelihood signal at low N - if a param's
    posterior sits near the PRIOR's own center (link-space 0, i.e. natural-
    scale bounds midpoint) regardless of the true data, that is the
    signature to check for before concluding "not identifiable."
    """
    if linked:
        spec = GENERIC_LINKSPACE_GROUP_PRIOR
        if mu_sigma is not None:
            spec = dict(spec)
            spec["mu"] = {"dist": "Normal", "mu": 0.0, "sigma": mu_sigma}
    else:
        if param not in HDDM_SETTINGS_GROUP:
            raise KeyError(
                f"no manual HDDM prior recorded for {param!r} in HDDM_SETTINGS_GROUP - "
                "add it there rather than falling back to get_hddm_default_prior."
            )
        spec = HDDM_SETTINGS_GROUP[param]
    prior = _build_prior(spec, bounds=None)
    if verbose:
        print(f"  [hssm_run_defaults] manual prior for {param!r} (linked={linked}): {prior}")
    return prior


# --------------------------------------------------------------------------- #
# arviz-version-proof summary and trace plot.
# arviz 1.x renamed summary's interval columns (hdi_3%/hdi_97% -> hdi94_lb/hdi94_ub, default
# now an 89% ETI) and rebuilt plot_trace on arviz-plots, which no longer takes figsize/compact
# and no longer returns an axes array. The tutorials index those columns and draw truth lines on
# those axes, so both are wrapped here.
# --------------------------------------------------------------------------- #
def summarize(post, **kw):
    """az.summary with a 94% HDI reported as hdi_3% / hdi_97%, on arviz 0.x and 1.x alike."""
    import inspect
    import arviz as az
    if "ci_kind" in inspect.signature(az.summary).parameters:          # arviz >= 1.0
        s = az.summary(post, ci_kind="hdi", ci_prob=0.94, **kw)
        return s.rename(columns={"hdi94_lb": "hdi_3%", "hdi94_ub": "hdi_97%"})
    return az.summary(post, hdi_prob=0.94, **kw)                          # arviz 0.x


def trace_plot(post, var_names, truth=None, title="", figsize=None):
    """One row per variable: left, per-chain density; right, draws per chain.

    `truth` maps a variable name to a scalar or an array of values; each is drawn dashed on
    both panels (thin and faint when there are many, as for per-subject effects). Variables
    with extra dimensions (e.g. 12 subjects) are overlaid in one row, coloured by subject.
    """
    import matplotlib.pyplot as plt
    n = len(var_names)
    fig, axes = plt.subplots(n, 2, figsize=figsize or (11, 2.0 * n), squeeze=False)
    for row, name in zip(axes, var_names):
        arr = np.asarray(post[name])
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)                  # chain, draw, extra
        many = arr.shape[2] > 1
        for c in range(arr.shape[0]):
            for k in range(arr.shape[2]):
                x = arr[c, :, k]
                color, alpha = (f"C{k % 10}", 0.45) if many else (f"C{c}", 0.9)
                lo, hi = np.min(x), np.max(x)
                if hi > lo:
                    grid = np.linspace(lo, hi, 200)
                    try:
                        from scipy.stats import gaussian_kde
                        row[0].plot(grid, gaussian_kde(x)(grid), lw=0.9, alpha=alpha, color=color)
                    except Exception:
                        row[0].hist(x, bins=40, histtype="step", density=True, alpha=alpha, color=color)
                row[1].plot(x, lw=0.5, alpha=alpha, color=color)
        for ax in row:
            ax.set_title(name, fontsize=10, loc="left")
        if truth is not None and name in truth:
            vals = np.atleast_1d(truth[name])
            lw, al = (0.6, 0.35) if vals.size > 1 else (1.2, 1.0)
            for tv in vals:
                row[0].axvline(tv, color="k", ls="--", lw=lw, alpha=al)
                row[1].axhline(tv, color="k", ls="--", lw=lw, alpha=al)
    if title:
        fig.suptitle(title, fontsize=13, y=1.002)
    fig.tight_layout()
    plt.show()
    return axes
