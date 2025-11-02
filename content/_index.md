+++
title = "Symmetry as Intervention;<br>Causal Estimation with Data Augmentation"
[extra]
authors = [
    {name = "Uzair Akbar", url = "https://uzairakbar.com"},
    {name = "Niki Kilbertus", url = "http://nikikilbertus.info/"},
    {name = "Hao Shen", url = "https://scholar.google.com/citations?hl=en&user=Kce9W-8AAAAJ"},
    {name = "Krikamol Muandet", url = "https://www.krikamol.org"},
    {name = "Bo Dai", url = "https://bo-dai.github.io/"},
]
venue = {name = "NeurIPS", date = 2025-12-02, url = "https://neurips.cc/virtual/2025/loc/san-diego/poster/119327"}
buttons = [
    {name = "Paper", url = "https://openreview.net/forum?id=C1LVIInfZO"},
    {name = "PDF", url = "https://arxiv.org/pdf/2510.25128"},
    {name = "Code", url = "https://github.com/uzairakbar/causal-data-augmentation"},
    {name = "Poster", url = "https://neurips.cc/virtual/2025/loc/san-diego/poster/119327"},
    {name = "Video", url = "https://neurips.cc/virtual/2025/loc/san-diego/poster/119327"},
]
katex = true
large_card = true
favicon = true
+++

**TLDR**: We show that data augmentation (DA) transformations that correspond to symmetries in data generaiton are equivalent to causal interventions. Such DA can hence be used to mitigate bias due to hidden confounding in observational `$(X, Y)$` data, improving causal estimation and robust prediction.

---

## Statistical vs. Causal Estimation
**Empirical risk minimization**: We are given `$n$` samples `$\mathcal{D}\coloneqq \{ (\mathbf{x}_i, \mathbf{y}_i) \}_{i=0}^n$` of outcome `$Y$` and treatment `$X$` from the following data generating process.
```
$$
Y = f(X) + \xi
\qquad
\mathbb{E}[\xi] = 0 .
\tag{1}
$$
```
We would like to estimate the function `$f$`. Under the standard assumption that noise `$\xi$` is uncorrelated with `$X$`, we can recover `$f$` via *empirical risk minimization (ERM)* by finding a function `$\widehat{h}_{\text{ERM}}$` that best predicts `$Y$` values for unlabelled `$X$` values. Regularization techniques like *data augmentation (DA)* are then used to reduce estimation variance by multiple random perturbations `$(G\mathbf{x}_i, \mathbf{y}_i)$` for each sample `$(\mathbf{x}_i, \mathbf{y}_i)\in\mathcal{D}$` using random transformation `$G$`.

**Causal Estimation**: However, `$X$` is generally correlated with `$\xi$`, rendering ERM based estimation biased. Known as *confounding bias*, it arises due to unobserved common causes `$C$` of `$X$` and `$Y$`, known as *confounders*. Removing confounding bias requires that we make `$X$` and `$\xi$` uncorrelated via an *intervention* where we independently assign values of `$X$` during data generation. Alas, interventions are often inaccessible compared to observational data.

A common workaround is to use auxiliary variables to correct for confounding. One approach is that of *instrumental variable (IV) regression*, where an instrument `$Z$` satisfies certain conditional independences with respect to `$(X, Y, C)$`. Unfortunately, even IVs are unavailable in many application domains.

## Causal Estimation with Data Augmentation
{% figure(
    src=["augmentation-intervention.svg"],
    alt=["Augmentation, intervention equivalence."],
    dark_invert=[true]
) %}
**Figure 1:** Graphs for the data generating process; *(left)* The original model `$(\dagger)$` post DA application. *(right)* The modified model `$(\ddagger)$` post soft-intervention.
{% end %}

**Outcome Invariant DA**: We consider DA transformations with respect to which `$f$` is invariant. Specifically, `$G$` takes values in `$\mathcal{G}$` such that `$f$` is *`$\mathcal{G}$`-invariant*:
```
$$
f(\mathbf{x}) = f(\mathbf{g} \mathbf{x}),
\qquad
\forall
\;\;
(\mathbf{x}, \mathbf{g})\in \mathcal{X}\times\mathcal{G}.
$$
```
Of course, constructing such DA requires knowledge of symmetries of `$f$`. For example, when classifying images of cats vs. dogs, the true labeling function would certainly be invariant to image rotations. `$G$` would then represent the random rotation angle, whereas `$G\mathbf{x}$` would be the rotated image `$\mathbf{x}$`.

**DA as soft-intervention**: Our key insight is that such DA on observatinoal `$(X, Y)$` data is equivalent to changing the data generating process `$(\dagger)$` itself to
```
$$
Y = f(GX) + \xi
\qquad
\mathbb{E}[\xi] = 0 .
\tag{2}
$$
```
DA therefore constitutes as a *soft-intervention* on `$X$`, and as such can mitigate hidden confounding bias, thereby improving causal estimation of `$f$`.

**DA as relaxed IVs**: Next, we frame DA transformations `$G$` as *IV-like (IVL)* by observing that they satisfy similar conditional independences as IVs by  construction. As such, we can use DA in composition with IV regression to further reduce confounding bias and improve causal estimation beyond simple DA.

## Robust Prediction
Reducing confounding bias, even when `$f$` itself may not be identifiable, is an upstream problem for *robust prediction*---predictors that generalize well to *out-of-distribution (OOD)* shifts in `$X$`. A predictor that fails on shifted distributions does so because it learned spurious correlations (i.e., confounding). We tackle this root cause directly by re-purposing the common  IID generalizaiton tool of DA to instead achieve downstream goals of OOD generalization.

## Experimental Results
Estimation error is captured in an interpretable way using *normalized causal excess risk (nCER)*---it is `$0$` for the true solution `$f$` and `$1$` for pure confounding.

{% figure(
    src=["sweep_plots.svg"],
    alt=["Linear Gaussian simulation experiment."],
    dark_invert=[true]
) %}
**Figure 2:** Simulation experiment for a linear Gaussian data generation model. `$\kappa$` and `$\gamma$` control the amount of confounding and *strength* of DA respectively. `$\alpha$` is the IVL regularization parameter. All three are set to `$1$` by default. Each data-point averages `$\operatorname{nCER}$` over `$32$` trials with a `$95\%$` confidence interval.
{% end %}

{% figure(
    src=["box_plots.svg"],
    alt=["Benchmark comparison with OOD baselines."],
    dark_invert=[true]
) %}
**Figure 3:** Experiment results; common OOD generalisation benchmarks compared against the ERM, DA+ERM and DA+IV baselines, including DA+IVL.
{% end %}

## Citation

```bibtex
@misc{akbar2025symmetryAsIntervention,
      title={An Analysis of Causal Effect Estimation using Outcome Invariant Data Augmentation}, 
      author={Uzair Akbar and Niki Kilbertus and Hao Shen and Krikamol Muandet and Bo Dai},
      year={2025},
      eprint={2510.25128},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.25128},
}
```
