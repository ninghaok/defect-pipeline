# Cost and capped 5-pixel tolerance protocol

Current, human-readable baseline: [BASELINE.md](../BASELINE.md).

Model promotion uses only paired, reviewed shadow samples containing OK and NG.
Every threshold-calibrated, smoke-tested candidate enters shadow without an offline
promotion comparison. The calibration set sets thresholds; the independent test
remains report-only.

A shadow candidate passes when `2*FN+FP` decreases, when cost ties and T5 strictly
increases, or when T5 increases by at least 20 percentage points regardless of cost.
There is no additional per-metric veto. All component metrics remain visible.

For ground truth G and prediction P, clip both to the original-resolution ROI,
then compute `A = |G|`, `M5 = |G \ D5(P)|`, and `E5 = |P \ D5(G)|`.
D5 is binary dilation using an 11-by-11 square, with zero outside the image.
The single-image score is `T5 = (A - M5) / (A + min(E5, A))`.
Cap excess separately for each image, then aggregate as
`sum(tolerant_matched_pixels) / sum(tolerant_total_pixels)`.
These fields store `A-M5` and `A+min(E5,A)`, respectively. A fully covered
defect with severe over-segmentation scores 0.5; complete misses score zero.
Additional excess beyond the cap is intentionally indistinguishable in T5;
IoU continues to report the extent of over-segmentation.
Only valid true-NG images enter this localization score. An image classified OK
uses an empty prediction, so a completely missed defect gets zero. OK false
alarms are represented by error cost and FPR. Scores do not average image/batch
percentages. IoU is retained unchanged.

Metric schema is 4, metric version is `tolerant_agreement_r5_chebyshev_capped_v2`,
and the rule ID is `shadow_cost_t5_capped_gain20_v3`.
Cached score identities include the metric version. Incompatible counts must
be recomputed from saved masks; they are never converted from an old percentage.
Equal-cost and 20-point gain comparisons use integer cross-products of the
numerators and denominators, without rounding or an epsilon.

New-rule experiments replay the frozen original stream from initialization,
with fresh state and output directories. Reusing later batches from an old-rule
trajectory would mix incompatible model, review and training histories.
The frozen October 8 run retains its actual decisions and original evidence;
rescoring its saved masks does not retroactively change promotions.
