"""
lib_est.py — estimation primitives, implemented in numpy only.

Everything the revision needs is here: penalised logistic regression by IRLS,
a ridge-penalised Cox model for the censoring process, weighted Kaplan-Meier
(with fixed or time-varying weights), restricted mean survival time, inverse
probability of censoring weights conditional on baseline covariates,
cross-fitted AIPW, and a patient-level bootstrap.

Censoring model. The identification assumption is that censoring is independent
of survival given treatment and the baseline covariates, C _||_ T | A, X. The
default censoring model implements exactly that: a ridge-penalised Cox model
for the censoring hazard, stratified by treatment arm (separate baseline per
arm, shared coefficients) with the adjustment covariates as regressors. The
arm-only Kaplan-Meier censoring model, which assumes C _||_ T | A alone, is kept
as an option (`censoring="km"`) so that the two can be reported side by side.

Why hand-rolled rather than scikit-learn / lifelines: the analysis environment
has neither, and at 12-26 events per arm a flexible ML nuisance model is not
defensible anyway. Every nuisance model here is a penalised linear or logistic
fit whose degrees of freedom are visible and small.

Conventions used throughout:
    time   -- follow-up in days from the landmark
    event  -- 1 death, 0 censored
    A      -- binary treatment
    tau    -- RMST horizon in days
"""

from __future__ import annotations

import numpy as np

# ── Design matrices ───────────────────────────────────────────────────────────

def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(len(X)), X])


def standardize(X: np.ndarray, mu=None, sd=None):
    """Centre/scale columns. Returns (Xs, mu, sd) so test folds reuse train stats."""
    X = np.asarray(X, dtype=float)
    if mu is None:
        mu = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def drop_constant(X: np.ndarray, keep=None):
    """Drop zero-variance columns (they make the ridge solve rank-deficient)."""
    X = np.asarray(X, dtype=float)
    if keep is None:
        keep = X.std(axis=0) > 1e-12
    return X[:, keep], keep


# ── Penalised logistic regression (IRLS) ──────────────────────────────────────

def logit_fit(X, y, ridge=1e-2, max_iter=60, tol=1e-8):
    """Ridge-penalised logistic regression by iteratively reweighted least squares.

    The intercept is not penalised. Returns the coefficient vector for a design
    matrix that already carries an intercept column.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    P = np.eye(p) * ridge
    P[0, 0] = 0.0  # never penalise the intercept

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-6, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        A = XtW @ X + P
        b = XtW @ z
        try:
            new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.all(np.isfinite(new)):
            break
        step = np.max(np.abs(new - beta))
        beta = new
        if step < tol:
            break
    return beta


def logit_predict(X, beta):
    eta = np.clip(np.asarray(X, dtype=float) @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


def wls_fit(X, y, w=None, ridge=1e-2):
    """Weighted ridge least squares. Intercept (column 0) unpenalised."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    P = np.eye(X.shape[1]) * ridge
    P[0, 0] = 0.0
    XtW = X.T * w
    A = XtW @ X + P
    b = XtW @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


# ── Kaplan-Meier and RMST ─────────────────────────────────────────────────────

def km_estimate(time, event, weights=None):
    """Weighted Kaplan-Meier. Returns (times, survival) including t=0, S=1."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    w = np.ones(len(time)) if weights is None else np.asarray(weights, dtype=float)

    if len(time) == 0:
        return np.array([0.0]), np.array([1.0])

    order = np.argsort(time, kind="mergesort")
    t, e, w = time[order], event[order], w[order]

    # Vectorised: at-risk weight is a reverse cumulative sum, and deaths per
    # distinct time are a block reduction. The naive loop is O(n^2), which makes
    # a 2000-replicate bootstrap impractical on the full cohort.
    uniq, first = np.unique(t, return_index=True)
    rev_cum = np.cumsum(w[::-1])[::-1]        # rev_cum[i] = sum of w[i:]
    at_risk = rev_cum[first]
    deaths = np.add.reduceat(w * e, first)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(at_risk > 0, deaths / at_risk, 0.0)
    surv = np.cumprod(1.0 - np.clip(frac, 0.0, 1.0))

    return (np.concatenate([[0.0], uniq.astype(float)]),
            np.concatenate([[1.0], surv]))


def rmst(times, surv, tau):
    """Area under the survival curve up to tau (step-function integration)."""
    times = np.asarray(times, dtype=float)
    surv = np.asarray(surv, dtype=float)
    keep = times <= tau
    t = np.concatenate([times[keep], [tau]])
    s = np.concatenate([surv[keep], [surv[keep][-1] if keep.any() else 1.0]])
    # S is right-continuous: on [t_j, t_{j+1}) the survival is s_j
    return float(np.sum(s[:-1] * np.diff(t)))


def km_rmst(time, event, tau, weights=None):
    t, s = km_estimate(time, event, weights)
    return rmst(t, s, tau)


def median_followup(time, event):
    """Reverse Kaplan-Meier: median follow-up treating censoring as the event."""
    t, s = km_estimate(time, 1 - np.asarray(event, dtype=int))
    below = np.where(s <= 0.5)[0]
    return float(t[below[0]]) if len(below) else float("nan")


# ── Cox proportional hazards (used for the censoring process) ─────────────────

def cox_fit(X, time, event, ridge=1.0, strata=None, beta0=None, max_iter=25,
            tol=1e-6):
    """Ridge-penalised Cox partial likelihood by Newton-Raphson, Breslow ties.

    `strata` gives a separate baseline hazard per stratum with coefficients
    shared across strata. X must be standardized and must not carry an
    intercept (the Cox model has none). The penalty is 0.5 * ridge * ||beta||^2.

    Risk-set sums are accumulated with the usual cumulative-sum identity, so a
    fit costs O(n p^2) per iteration and no n x p x p array is formed.

    Returns dict(beta, base={stratum: (unique_times, H0)}, converged, n_iter,
    loglik), where H0 is the Breslow cumulative baseline hazard at the unique
    times of that stratum.
    """
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    n, p = X.shape
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)

    groups = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        order = idx[np.argsort(time[idx], kind="mergesort")]
        t_s = time[order]
        uniq, first, inv = np.unique(t_s, return_index=True, return_inverse=True)
        e_s = event[order].astype(float)
        groups.append(dict(s=s, uniq=uniq, first=first, inv=inv, e=e_s,
                           d=np.add.reduceat(e_s, first), X=X[order],
                           xsum_events=(X[order] * e_s[:, None]).sum(axis=0)))

    # A vector permits unpenalized treatment in the Cox outcome benchmark.
    penalty = np.broadcast_to(np.asarray(ridge, dtype=float), (p,))
    P = np.diag(penalty)
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float)

    def loglik(b):
        ll = -0.5 * float(b @ P @ b)
        for G in groups:
            eta = np.clip(G["X"] @ b, -30, 30)
            w = np.exp(eta)
            S0 = np.cumsum(w[::-1])[::-1][G["first"]]
            ll += float((eta * G["e"]).sum() - (G["d"] * np.log(S0)).sum())
        return ll

    ll_old = loglik(beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        g = -P @ beta
        H = P.copy()
        for G in groups:
            eta = np.clip(G["X"] @ beta, -30, 30)
            w = np.exp(eta)
            S0 = np.cumsum(w[::-1])[::-1][G["first"]]                # per unique time
            S1 = np.cumsum((w[:, None] * G["X"])[::-1], axis=0)[::-1][G["first"]]
            M = S1 / S0[:, None]                                    # S1/S0 per unique time
            c = np.cumsum(G["d"] / S0)[G["inv"]]                    # sum_{u<=t_j} d_u/S0_u
            g += G["xsum_events"] - (M * G["d"][:, None]).sum(axis=0)
            H += (G["X"] * (w * c)[:, None]).T @ G["X"] - (M * G["d"][:, None]).T @ M
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        # Step halving guards the rare replicate where a full Newton step
        # overshoots. The penalised partial likelihood is concave, so a short
        # enough step always improves it.
        t_step = 1.0
        cand, ll_new = beta, ll_old
        for _ in range(6):
            cand = beta + t_step * step
            ll_new = loglik(cand)
            if np.isfinite(ll_new) and ll_new >= ll_old - 1e-10:
                break
            t_step *= 0.5
        moved = float(np.max(np.abs(cand - beta)))
        beta, ll_old = cand, ll_new
        if moved < tol:
            converged = True
            break

    base = {}
    for G in groups:
        w = np.exp(np.clip(G["X"] @ beta, -30, 30))
        S0 = np.cumsum(w[::-1])[::-1][G["first"]]
        base[G["s"]] = (G["uniq"], np.cumsum(G["d"] / S0))
    return dict(beta=beta, base=base, converged=converged, n_iter=it,
                loglik=ll_old)


# ── Inverse probability of censoring weights ──────────────────────────────────

def censoring_survival(time, event, strata=None):
    """Return a callable G(t) estimating P(not yet censored at t-).

    Arm-only model: Kaplan-Meier on the censoring indicator, one curve per
    stratum (here: treatment arm). This implements C _||_ T | A. It is retained
    for the sensitivity analysis and the lifelines cross-check; the default
    model conditional on covariates is `CensoringModel`.
    """
    time = np.asarray(time, dtype=float)
    cens = 1 - np.asarray(event, dtype=int)

    if strata is None:
        strata = np.zeros(len(time), dtype=int)
    strata = np.asarray(strata)

    curves = {}
    for s in np.unique(strata):
        m = strata == s
        curves[s] = km_estimate(time[m], cens[m])

    def G(t_query, s_query=None):
        t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
        if s_query is None:
            s_query = np.zeros(len(t_query), dtype=int)
        s_query = np.atleast_1d(np.asarray(s_query))
        out = np.ones(len(t_query))
        for s in np.unique(s_query):
            m = s_query == s
            ct, cs = curves[s]
            # step just before t: largest curve time strictly less than t
            idx = np.searchsorted(ct, t_query[m], side="left") - 1
            idx = np.clip(idx, 0, len(cs) - 1)
            out[m] = cs[idx]
        return np.clip(out, 1e-3, 1.0)

    return G


def _left_limit_H0(uniq, H0, t):
    """Cumulative baseline hazard just before each time in t."""
    if len(H0) == 0:
        return np.zeros(len(t))
    idx = np.searchsorted(uniq, t, side="left") - 1
    return np.where(idx >= 0, H0[np.clip(idx, 0, len(H0) - 1)], 0.0)


class CensoringModel:
    """Censoring survival function G(t | A, X), evaluated as a left limit.

    kind="cox"  ridge-penalised Cox model for the censoring hazard, stratified
                by arm with coefficients shared across arms, regressors X.
                Implements C _||_ T | A, X, the assumption stated in the
                identification section.
    kind="km"   arm-stratified Kaplan-Meier. Implements C _||_ T | A only.

    X is standardized inside `fit` and the same centring is applied at
    prediction time, so a model fitted on a training fold predicts correctly
    for a held-out fold.
    """

    def __init__(self, kind="cox"):
        if kind not in ("cox", "km"):
            raise ValueError(f"unknown censoring model {kind!r}")
        self.kind = kind

    def fit(self, time, event, A, X=None, ridge=None, ridge_scale=1.0):
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=int)
        cens = 1 - event
        A = np.asarray(A, dtype=int)
        if self.kind == "km" or X is None:
            self.kind_used = "km"
            self.G_km = censoring_survival(time, event, strata=A)
            self.curves = {a: km_estimate(time[A == a], cens[A == a])
                           for a in np.unique(A)}
            self.converged = True
            self.beta_max = 0.0
            self.ridge = 0.0
            return self
        Xc, keep = drop_constant(np.asarray(X, dtype=float))
        Xs, mu, sd = standardize(Xc)
        n, p = Xs.shape
        lam = (adaptive_ridge(n, p + 1) if ridge is None else float(ridge)) * ridge_scale
        f = cox_fit(Xs, time, cens, ridge=lam, strata=A)
        self.kind_used = "cox"
        self.keep, self.mu, self.sd = keep, mu, sd
        self.beta, self.base, self.ridge = f["beta"], f["base"], lam
        self.converged = bool(f["converged"])
        self.beta_max = float(np.max(np.abs(f["beta"]))) if p else 0.0
        return self

    def _risk(self, X):
        Xs = (np.asarray(X, dtype=float)[:, self.keep] - self.mu) / self.sd
        return np.exp(np.clip(Xs @ self.beta, -30, 30))

    def G(self, t, A, X=None):
        """G(t- | A, X) for each subject at that subject's own time t."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        A = np.atleast_1d(np.asarray(A, dtype=int))
        if self.kind_used == "km":
            return self.G_km(t, A)
        out = np.ones(len(t))
        risk = self._risk(X)
        for a in np.unique(A):
            m = A == a
            uniq, H0 = self.base.get(a, (np.array([]), np.array([])))
            out[m] = np.exp(-_left_limit_H0(uniq, H0, t[m]) * risk[m])
        return np.clip(out, 1e-3, 1.0)

    def curve(self, a, X, grid):
        """G(grid- | A=a, X_i): one row per subject, one column per grid time."""
        grid = np.asarray(grid, dtype=float)
        if self.kind_used == "km":
            ct, cs = self.curves[a]
            idx = np.clip(np.searchsorted(ct, grid, side="left") - 1, 0, len(cs) - 1)
            row = np.clip(cs[idx], 1e-3, 1.0)
            n_rows = len(np.atleast_2d(X)) if X is not None else 1
            return np.tile(row, (n_rows, 1))
        uniq, H0 = self.base.get(a, (np.array([]), np.array([])))
        h = _left_limit_H0(uniq, H0, grid)
        return np.clip(np.exp(-np.outer(self._risk(X), h)), 1e-3, 1.0)


G_FLOOR = 0.05          # censoring survival is not trusted below this
WEIGHT_CAP_Q = 0.99     # combined weights truncated at this quantile


def restricted_outcome(time, event, tau):
    """Y = min(time, tau) and the indicator that Y is fully observed."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    Y = np.minimum(time, tau)
    complete = ((event == 1) & (time <= tau)) | (time >= tau)
    return Y, complete.astype(int)


def ipcw_components(time, event, tau, strata=None, g_floor=G_FLOOR):
    """Restricted time, completeness indicator and IPC weight for RMST under
    the arm-only Kaplan-Meier censoring model.

    Y     = min(time, tau)
    Delta = 1 if the patient's restricted time is fully observed
            (died at or before tau, or followed at least to tau)
    w     = Delta / G(Y)   -- zero for patients censored before tau

    Used where only Y and Delta are needed (the diagnostic rho, the nuisance
    budget, the minimum detectable effect). The estimators fit their own
    censoring model, conditional on covariates by default; see aipw_rmst and
    ipcw_km_rmst_contrast.

    G is floored at `g_floor`, capping any single IPC weight at 1/g_floor = 20.
    Without a floor the weight is unbounded: in this cohort the censoring curve
    falls low well before tau (median follow-up 974 days against a 1095-day
    primary horizon), and a handful of patients in the far tail would otherwise
    dominate the estimate. The floor introduces a known, bounded bias towards
    the null in exchange for a finite variance; the number of observations it
    binds on is reported as a diagnostic.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    Y, complete = restricted_outcome(time, event, tau)
    G = censoring_survival(time, event, strata)
    g_raw = G(Y, strata if strata is not None else None)
    g = np.maximum(g_raw, g_floor)
    w = np.where(complete == 1, 1.0 / g, 0.0)
    return Y, complete, w, g_raw


# ── Diagnostics ───────────────────────────────────────────────────────────────

def smd(X, A, w=None):
    """Standardized mean difference per column, optionally weighted."""
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=int)
    w = np.ones(len(A)) if w is None else np.asarray(w, dtype=float)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        x = X[:, j]
        m1 = np.average(x[A == 1], weights=w[A == 1])
        m0 = np.average(x[A == 0], weights=w[A == 0])
        v1 = np.average((x[A == 1] - m1) ** 2, weights=w[A == 1])
        v0 = np.average((x[A == 0] - m0) ** 2, weights=w[A == 0])
        pooled = np.sqrt((v1 + v0) / 2.0)
        out[j] = 0.0 if pooled < 1e-12 else (m1 - m0) / pooled
    return out


def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))


def ridge_effective_df(D, w, ridge):
    """Effective degrees of freedom of a weighted ridge regression,
    tr(D (D'WD + P)^{-1} D'W), with the intercept (column 0) unpenalised.

    The nominal parameter count p overstates the complexity of a penalised fit;
    this is the quantity that would replace p in the numerator of rho if the
    penalty were taken into account (Hastie, Tibshirani and Friedman, 2009,
    Section 3.4.1). Reported alongside p, not used in place of it.
    """
    D = np.asarray(D, dtype=float)
    w = np.ones(len(D)) if w is None else np.asarray(w, dtype=float)
    P = np.eye(D.shape[1]) * float(ridge)
    P[0, 0] = 0.0
    DtW = D.T * w
    A = DtW @ D + P
    try:
        return float(np.trace(np.linalg.solve(A, DtW @ D)))
    except np.linalg.LinAlgError:
        return float("nan")


# ── Cross-fitted AIPW for the RMST difference ─────────────────────────────────

def _folds(n, k, rng, strata=None):
    """Cross-fitting folds, stratified when `strata` is supplied.

    Plain random assignment is not safe here. With 16 events split over five
    folds, a fold can hold one or two events, the outcome model for that arm
    falls back to an arm mean, and the fallback is invisible in the result.
    Stratifying jointly on treatment and event keeps each fold's composition
    close to the sample's.
    """
    if strata is None:
        return np.array_split(rng.permutation(n), k)

    strata = np.asarray(strata)
    buckets = [[] for _ in range(k)]
    for s in np.unique(strata):
        members = rng.permutation(np.where(strata == s)[0])
        # deal round-robin from a rotating start so no fold is favoured
        start = int(rng.integers(0, k))
        for j, idx in enumerate(members):
            buckets[(start + j) % k].append(idx)
    return [np.asarray(b, dtype=int) for b in buckets]


def adaptive_ridge(n, p, base=1.0):
    """Ridge strength for a nuisance model with p covariates and n observations.

    At the sample sizes in this cohort (n as low as 136 with 16 events against
    adjustment sets of up to ~50 variables) an unpenalised logistic propensity
    model separates, driving fitted probabilities to 0 or 1 and the inverse
    weights to hundreds. Scaling the penalty with p/n keeps the fitted
    propensity inside the range the data can support. On standardized
    covariates a penalty of order p is mild but sufficient.
    """
    return float(max(base, p * max(1.0, 10.0 * p / max(n, 1))))


def aipw_rmst(time, event, A, X, tau, k=5, seed=0, trim=0.0,
              ridge=None, ps_clip=(0.02, 0.98), return_psi=False,
              weight_cap_q=WEIGHT_CAP_Q, g_floor=G_FLOOR, clip_m=True,
              clip_estimate=True, censoring="cox", ridge_scale=1.0):
    """Cross-fitted AIPW estimate of E[min(T,tau)|do(A=1)] - E[...|do(A=0)].

    Nuisances, all fitted out-of-fold:
      e(X)      propensity, ridge logistic (penalty from adaptive_ridge)
      G(t|A,X)  censoring survival: arm-stratified ridge Cox on X
                (censoring="cox", the default) or arm-stratified Kaplan-Meier
                (censoring="km"); floored at g_floor
      m_a(X)    IPCW-weighted ridge regression of min(T,tau) on X within arm a

    `ridge_scale` multiplies every adaptive penalty; it exists for the
    penalty-sensitivity analysis of the diagnostic and is 1 in the analysis.

    Stabilisation, all reported rather than hidden:
      * fitted propensities clipped to `ps_clip`
      * censoring survival floored at g_floor
      * the combined weight  1/e(X) or 1/(1-e(X))  times the IPC weight is
        truncated at its `weight_cap_q` quantile
      * the estimate is clipped to [-tau, tau], which is the range the estimand
        can occupy by definition; if the raw value falls outside, that is
        recorded in `out_of_range`

    Returns a dict. `psi` is the per-patient contribution; its mean is the point
    estimate. Intervals come from the bootstrap, never from the spread of psi.
    """
    rng = np.random.default_rng(seed)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    A = np.asarray(A, dtype=int)
    X = np.asarray(X, dtype=float)
    n = len(A)

    Y, complete = restricted_outcome(time, event, tau)

    Xc, keep = drop_constant(X)
    Xs, mu, sd = standardize(Xc)
    D = add_intercept(Xs)
    p = D.shape[1]
    lam = (adaptive_ridge(n, p) if ridge is None else float(ridge)) * ridge_scale

    kk = int(min(k, max(2, (A == 1).sum(), (A == 0).sum())))
    kk = max(2, min(kk, n // 10 if n >= 20 else 2))
    # Stratify jointly on treatment and the observed event indicator so that no
    # fold is left without events in an arm.
    strata = A.astype(int) * 2 + (event.astype(int) & (time <= tau).astype(int))
    folds = _folds(n, kk, rng, strata=strata)

    ps = np.zeros(n)
    m1 = np.zeros(n)
    m0 = np.zeros(n)
    g_raw = np.ones(n)
    fb_propensity = 0      # folds where the propensity model could not be fitted
    fb_outcome = 0         # arm-folds where the outcome model fell back to a mean
    cens_converged = True
    cens_beta_max = 0.0
    fit_rows = []

    for f in folds:
        te = np.zeros(n, dtype=bool)
        te[f] = True
        tr = ~te
        if tr.sum() < 10 or len(np.unique(A[tr])) < 2:
            fb_propensity += 1
            fb_outcome += 2
            ps[te] = A.mean()
            m1[te] = Y[complete == 1].mean() if complete.any() else tau
            m0[te] = m1[te]
            continue

        # Censoring model on the training rows, evaluated for the training
        # rows (to weight the outcome regressions) and for the held-out rows.
        cm = CensoringModel(censoring).fit(time[tr], event[tr], A[tr],
                                          X=Xc[tr], ridge_scale=ridge_scale)
        cens_converged &= cm.converged
        cens_beta_max = max(cens_beta_max, cm.beta_max)
        g_tr = np.maximum(cm.G(Y[tr], A[tr], Xc[tr]), g_floor)
        g_raw[te] = cm.G(Y[te], A[te], Xc[te])
        w_tr = np.where(complete[tr] == 1, 1.0 / g_tr, 0.0)

        b = logit_fit(D[tr], A[tr], ridge=lam)
        ps[te] = logit_predict(D[te], b)

        for a, dest in ((1, m1), (0, m0)):
            sel_tr = (A[tr] == a) & (complete[tr] == 1)
            if sel_tr.sum() >= 5:
                coef = wls_fit(D[tr][sel_tr], Y[tr][sel_tr], w=w_tr[sel_tr],
                               ridge=lam)
                dest[te] = D[te] @ coef
                if return_psi:
                    fit_rows.append(dict(fold=len(fit_rows) // 2 + 1,
                                         treatment_arm=a, n_complete=int(sel_tr.sum()),
                                         n_parameters=p, ridge=lam,
                                         effective_df=ridge_effective_df(
                                             D[tr][sel_tr], w_tr[sel_tr], lam)))
            else:
                fb_outcome += 1
                dest[te] = (Y[tr][A[tr] == a].mean()
                            if (A[tr] == a).any() else tau)

    g = np.maximum(g_raw, g_floor)
    w_ipc = np.where(complete == 1, 1.0 / g, 0.0)

    ps = np.clip(ps, ps_clip[0], ps_clip[1])
    if clip_m:
        m1 = np.clip(m1, 0.0, tau)
        m0 = np.clip(m0, 0.0, tau)

    keep_mask = np.ones(n, dtype=bool)
    if trim > 0:
        keep_mask = (ps >= trim) & (ps <= 1 - trim)

    # Combined inverse weight, truncated at a stated quantile. Reported, not hidden.
    raw_w = np.where(A == 1, 1.0 / ps, 1.0 / (1.0 - ps)) * w_ipc
    pos = raw_w[raw_w > 0]
    cap = (float(np.quantile(pos, weight_cap_q))
           if (len(pos) and weight_cap_q < 1.0) else np.inf)
    n_capped = int((raw_w > cap).sum())
    cw = np.minimum(raw_w, cap)

    psi = (
        A * cw * (Y - m1) + m1
        - (1 - A) * cw * (Y - m0) - m0
    )
    raw_est = float(psi[keep_mask].mean()) if keep_mask.any() else float("nan")
    est = float(np.clip(raw_est, -tau, tau)) if clip_estimate else raw_est

    comp = complete == 1
    out = dict(
        estimate=est,
        estimate_raw=raw_est,
        out_of_range=bool(abs(raw_est) > tau),
        n=int(n),
        n_kept=int(keep_mask.sum()),
        n_trimmed=int((~keep_mask).sum()),
        n_treated=int((A == 1).sum()),
        n_control=int((A == 0).sum()),
        events=int(((event == 1) & (time <= tau)).sum()),
        events_treated=int(((event == 1) & (time <= tau) & (A == 1)).sum()),
        events_control=int(((event == 1) & (time <= tau) & (A == 0)).sum()),
        ps_min=float(ps.min()), ps_max=float(ps.max()),
        ridge=lam,
        fallback_propensity=int(fb_propensity),
        fallback_outcome=int(fb_outcome),
        min_complete_per_arm_fold=int(min(
            (int(((~te_) & (A == a) & (complete == 1)).sum())
             for te_ in [np.isin(np.arange(n), f) for f in folds]
             for a in (0, 1)), default=0)),
        weight_cap=cap, n_weights_capped=n_capped,
        weight_max=float(cw.max()),
        ess_treated=effective_sample_size(cw[A == 1]),
        ess_control=effective_sample_size(cw[A == 0]),
        folds=int(kk),
        # Censoring-weight diagnostics, evaluated at each complete case's own
        # restricted time.
        censoring_model=censoring,
        cens_G_min=float(g_raw[comp].min()) if comp.any() else float("nan"),
        cens_G_p05=float(np.percentile(g_raw[comp], 5)) if comp.any() else float("nan"),
        cens_n_below_floor=int((g_raw[comp] < g_floor).sum()),
        cens_ipc_weight_max=float(w_ipc.max()),
        cens_ipc_ess=effective_sample_size(w_ipc[comp]),
        cens_converged=bool(cens_converged),
        cens_beta_max=float(cens_beta_max),
    )
    if return_psi:
        out["psi"] = psi
        out["ps"] = ps
        out["keep"] = keep_mask
        out["cw"] = cw
        out["w_ipc"] = w_ipc
        out["outcome_fits"] = fit_rows
    return out


def km_estimate_tv(time, event, weights, grid, G):
    """Kaplan-Meier with time-varying inverse-probability-of-censoring weights.

    At each grid time t_j the at-risk mass is  sum_{i: Y_i >= t_j} w_i / G_ij
    and the event mass is  sum_{i: Y_i = t_j, delta_i = 1} w_i / G_ij, where
    G_ij = G(t_j- | A_i, X_i) is the subject's own censoring survival just
    before t_j. This is the IPCW Kaplan-Meier of Robins and Finkelstein (2000)
    with an additional fixed per-subject weight w_i (here the stabilised
    treatment weight). When G does not vary across subjects at a given time,
    every censoring weight in the ratio cancels and the estimator is the
    ordinary weighted Kaplan-Meier (Satten and Datta, 2001).

    Returns (times, survival) including t=0, S=1. `grid` must be the sorted
    unique observed times of this stratum and G an (n x len(grid)) array.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    w = np.asarray(weights, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if len(time) == 0:
        return np.array([0.0]), np.array([1.0])
    W = w[:, None] / G
    at_risk = ((time[:, None] >= grid[None, :]) * W).sum(axis=0)
    j = np.searchsorted(grid, time)
    ev = event == 1
    deaths = np.zeros(len(grid))
    np.add.at(deaths, j[ev], W[np.arange(len(time))[ev], j[ev]])
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(at_risk > 0, deaths / at_risk, 0.0)
    surv = np.cumprod(1.0 - np.clip(frac, 0.0, 1.0))
    return (np.concatenate([[0.0], grid]), np.concatenate([[1.0], surv]))


def ipcw_km_rmst_contrast(time, event, A, tau, X=None, seed=0, ridge=None,
                          ps_clip=(0.01, 0.99), trim=0.0, censoring="cox",
                          g_floor=G_FLOOR, ridge_scale=1.0, return_weights=False):
    """IPTW- and IPCW-weighted Kaplan-Meier contrast of RMST at tau.

    Treatment weighting is stabilised IPTW (propensity from a ridge logistic
    on X, or unweighted when X is None). Censoring is handled by the
    Kaplan-Meier product limit with time-varying weights 1/G(t- | A, X) from
    the arm-stratified ridge Cox censoring model (censoring="cox", default) or
    by the arm-stratified Kaplan-Meier (censoring="km"), in which case the
    censoring weights cancel and the estimator is the arm-wise IPTW-weighted
    Kaplan-Meier. The RMST is the area under each arm's curve to tau and the
    contrast is their difference.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    A = np.asarray(A, dtype=int)
    n = len(A)

    if X is None:
        ps = np.full(n, A.mean())
        Xc = None
    else:
        Xc, _ = drop_constant(np.asarray(X, dtype=float))
        Xs, _, _ = standardize(Xc)
        D = add_intercept(Xs)
        lam = (adaptive_ridge(n, D.shape[1]) if ridge is None else float(ridge)) * ridge_scale
        ps = logit_predict(D, logit_fit(D, A, ridge=lam))
    ps = np.clip(ps, ps_clip[0], ps_clip[1])

    keep = np.ones(n, dtype=bool)
    if trim > 0:
        keep = (ps >= trim) & (ps <= 1 - trim)

    p1 = A.mean()
    sw = np.where(A == 1, p1 / ps, (1 - p1) / (1 - ps))
    sw = sw * keep

    use_cox = censoring == "cox" and Xc is not None
    cm = CensoringModel("cox" if use_cox else "km").fit(
        time, event, A, X=Xc if use_cox else None, ridge_scale=ridge_scale)

    Y, complete = restricted_outcome(time, event, tau)
    g_own = cm.G(Y, A, Xc) if use_cox else cm.G(Y, A)

    r = {}
    for a in (1, 0):
        m = A == a
        if use_cox:
            grid = np.unique(time[m])
            Gm = np.maximum(cm.curve(a, Xc[m], grid), g_floor)
            t_a, s_a = km_estimate_tv(time[m], event[m], sw[m], grid, Gm)
            r[a] = rmst(t_a, s_a, tau)
        else:
            r[a] = km_rmst(time[m], event[m], tau, weights=sw[m])

    comp = complete == 1
    g_fl = np.maximum(g_own, g_floor)
    w_ipc = np.where(comp, 1.0 / g_fl, 0.0)
    result = dict(
        estimate=float(r[1] - r[0]),
        rmst_treated=float(r[1]),
        rmst_control=float(r[0]),
        n_kept=int(keep.sum()),
        n_trimmed=int((~keep).sum()),
        ess_treated=effective_sample_size(sw[A == 1]),
        ess_control=effective_sample_size(sw[A == 0]),
        censoring_model="cox" if use_cox else "km",
        cens_G_min=float(g_own[comp].min()) if comp.any() else float("nan"),
        cens_G_p05=float(np.percentile(g_own[comp], 5)) if comp.any() else float("nan"),
        cens_n_below_floor=int((g_own[comp] < g_floor).sum()),
        cens_ipc_weight_max=float(w_ipc.max()),
        cens_ipc_ess=effective_sample_size(w_ipc[comp]),
        cens_converged=bool(cm.converged),
        cens_beta_max=float(cm.beta_max),
    )
    if return_weights:
        result.update(ps=ps, treatment_weights=sw)
    return result


def cox_rmst_contrast(time, event, A, X, tau, ridge=None, return_fit=False):
    """Outcome Cox model, standardized to the empirical covariate distribution.

    Treatment has one unpenalized log hazard ratio; standardized covariates
    receive the same adaptive ridge rule as the primary estimator. Breslow
    ties/baseline; integrate the right-continuous predicted survival exactly
    over event-time intervals. Conditional independent censoring and a correct
    outcome model are required; this estimator is not doubly robust.
    """
    time, event, A = np.asarray(time, float), np.asarray(event, int), np.asarray(A, int)
    if len(np.unique(A)) != 2:
        raise ValueError("Cox contrast requires both treatment groups")
    Xc, keep = drop_constant(np.asarray(X, float))
    Xs, mu, sd = standardize(Xc)
    D = np.column_stack([A, Xs])
    lam = adaptive_ridge(len(A), Xs.shape[1] + 1) if ridge is None else float(ridge)
    f = cox_fit(D, time, event, ridge=np.r_[0.0, np.full(Xs.shape[1], lam)],
                max_iter=60, tol=1e-8)
    if not f['converged']:
        raise RuntimeError("Cox outcome model did not converge")
    times, H = f['base'][0]
    # Include zero and tau; an event exactly at tau has zero integration length.
    knots = np.unique(np.r_[0.0, times[(times > 0) & (times < tau)], float(tau)])
    idx = np.searchsorted(times, knots[:-1], side='right') - 1
    hazard = np.where(idx >= 0, H[np.maximum(idx, 0)], 0.0)
    duration = np.diff(knots)
    means = {}
    lp_x = Xs @ f['beta'][1:]
    for a in (0, 1):
        risks = np.exp(np.clip(lp_x + a * f['beta'][0], -30, 30))
        surv = np.exp(-risks[:, None] * hazard[None, :])
        means[a] = float(np.mean(surv @ duration))
    out = dict(estimate=means[1] - means[0], rmst_treated=means[1],
               rmst_control=means[0], treatment_log_hr=float(f['beta'][0]),
               treatment_hr=float(np.exp(f['beta'][0])), ridge=float(lam),
               n_parameters=D.shape[1], converged=f['converged'])
    if return_fit:
        out.update(fit=f, design=D, keep=keep, mean=mu, sd=sd)
    return out


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap(fn, n, B=2000, seed=0, alpha=0.05):
    """Patient-level nonparametric bootstrap of `fn(idx) -> float`.

    Replicates that fail (degenerate resample, one arm empty) are counted and
    reported rather than silently dropped.
    """
    rng = np.random.default_rng(seed)
    vals, failed = [], 0
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            v = fn(idx)
        except Exception:
            v = np.nan
        if v is None or not np.isfinite(v):
            failed += 1
        else:
            vals.append(float(v))
    vals = np.asarray(vals)
    if len(vals) == 0:
        return dict(lo=float("nan"), hi=float("nan"), se=float("nan"),
                    B_ok=0, B_failed=failed)
    return dict(
        lo=float(np.percentile(vals, 100 * alpha / 2)),
        hi=float(np.percentile(vals, 100 * (1 - alpha / 2))),
        se=float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
        B_ok=int(len(vals)),
        B_failed=int(failed),
    )


# ── Minimum detectable effect ─────────────────────────────────────────────────

def mde_rmst(sd_outcome, n1, n0, alpha=0.05, power=0.80):
    """Two-sided minimum detectable difference in means at the given power.

    z-based, since the analysis environment has no t/normal quantile function.
    z(0.975) = 1.959964, z(0.80) = 0.841621.
    """
    if n1 < 1 or n0 < 1:
        return float("nan")
    z_a, z_b = 1.959964, 0.841621
    return float((z_a + z_b) * sd_outcome * np.sqrt(1.0 / n1 + 1.0 / n0))


def mde_heterogeneity(events, sd_outcome, n1, n0, frac=0.5):
    """Minimum detectable *difference between two subgroups' effects*.

    Splitting a cohort into two subgroups of size `frac` and 1-frac and
    contrasting their treatment effects inflates the detectable difference by
    sqrt(1/frac + 1/(1-frac)) relative to the overall effect.
    """
    base = mde_rmst(sd_outcome, n1 * frac, n0 * frac)
    return float(base * np.sqrt(2.0))
