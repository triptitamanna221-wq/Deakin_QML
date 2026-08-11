"""Construct x_i in R^6 per BRIEF.md Sec 2.1: leverage, Tier-1 ratio,
CDS-proxy (or equity-vol proxy), liquidity ratio, out-degree centrality,
macro sensitivity.

FR Y-15 has no capital/equity or market-price data, so it cannot supply a
literal Tier-1 ratio, leverage ratio, CDS spread, or liquidity ratio. Four
of the six features below are therefore explicit PROXIES built from real
FR Y-15 indicator items chosen for plausible directional relevance, not
fabricated numbers standing in for the named quantity. This substitution is
disclosed here, in NOTES.md, and must be disclosed again in the paper
(BRIEF.md Sec 5 point 7: "if CDS data was unavailable, change the label
definition ... do not leave an unreproducible definition in the text" --
same principle applied to features).

  size_proxy         ("leverage" slot) = z(Y832), total exposure. Not a
                       true leverage ratio (no equity data); a size proxy.
  complexity_proxy   ("Tier-1" slot)   = z(M411 / Y832), OTC derivatives
                       notional over total exposure. Higher = more complex
                       trading book, one real driver of regulatory capital
                       stringency, but not capital adequacy itself.
  funding_proxy      ("CDS-proxy" slot) = z(M376 / Y832), securities
                       outstanding (wholesale market funding) over total
                       exposure. Higher wholesale-funding reliance is a
                       standard correlate of market-perceived credit risk
                       (CDS spreads), used here as a substitute.
  liquidity_proxy    (liquidity slot)  = z(M390 / Y832), payments activity
                       over total exposure, as a balance-sheet-turnover
                       proxy. Not a regulatory liquidity coverage ratio.
  out_degree         = real, from the reconstructed network W for that
                       snapshot: row i's share of total outgoing exposure.
  macro_breadth_proxy (macro-sensitivity slot) = z((M422+M426) / Y832),
                       cross-jurisdictional claims+liabilities over total
                       exposure, as a proxy for breadth of exposure to
                       global (hence macro) shocks. Not an estimated beta
                       against C_t (only 4 annual points exist; a
                       regression beta would be unreliably noisy).

Standardization: z-score using TRAINING-SPLIT statistics only, then
arctan, per BRIEF.md Sec 2.1.
"""
import numpy as np

FEATURE_NAMES = [
    "size_proxy", "complexity_proxy", "funding_proxy",
    "liquidity_proxy", "out_degree", "macro_breadth_proxy",
]


def raw_features(marginals_row, W_row_out_share):
    """marginals_row: dict/Series with y832, m376, m390, m411, m422, m426.
    W_row_out_share: this institution's share of total outgoing exposure in
    the snapshot's reconstructed network (real out-degree centrality)."""
    y832 = marginals_row["y832"]
    return np.array([
        y832,
        marginals_row["m411"] / y832,
        marginals_row["m376"] / y832,
        marginals_row["m390"] / y832,
        W_row_out_share,
        (marginals_row["m422"] + marginals_row["m426"]) / y832,
    ])


def build_raw_feature_panel(marginal_panel_df_by_t, W_panel):
    """marginal_panel_df_by_t: list of length T, each a DataFrame indexed by
    institution with columns y832,m376,m390,m411,m422,m426 (n rows).
    W_panel: (T, n, n). Returns raw feature array (T, n, 6), NOT yet
    standardized."""
    T, n, _ = W_panel.shape
    X = np.zeros((T, n, 6))
    for t in range(T):
        out_total = W_panel[t].sum(axis=1)
        out_share = np.divide(out_total, out_total.sum(), out=np.zeros(n), where=out_total.sum() > 0)
        df = marginal_panel_df_by_t[t]
        for i in range(n):
            X[t, i] = raw_features(df.iloc[i], out_share[i])
    return X


def standardize_train_split(X, train_end_t):
    """z-score using stats from t < train_end_t only, then arctan. X: (T,n,6)."""
    train = X[:train_end_t]
    mu = train.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = train.reshape(-1, X.shape[-1]).std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return np.arctan(Z), mu, sd
