# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-26

### Added

- Native `lcMethodKML` implementation for trajectory-vector clustering in Python.
- KML tuning controls: `mode` (`kml_fast`/`kml_strict`), `nStarts`, `nInit`, `maxIter`,
  `center`, `scale`, and `distance`.
- Strict-mode multi-start selection using a trajectory-centric within-cluster distance criterion.
- Deterministic relabeling of clusters for stable label ordering across runs.
- Parity-oriented test for R demo style K=4 behavior (ARI + class proportion tolerance).

## [0.1.0] - 2025-01-01

### Added

- Core framework: LCMethod, LCModel, LCModels base classes.
- Pipeline functions: latrendCluster, latrendBatchCluster, latrendRepCluster.
- Clustering methods: lcMethodRandom, lcMethodLMKM, lcMethodFeatures.
- Data utilities: generateTrajectories, generateLongData, latrendData.
- Wide/long format converters: tsmatrix, tsframe.
- Plotting with R-matching ggplot2 theme: plotTrajectories, plotClusterTrajectories,
  plotFittedTrajectories, plotMetric, plotClassProportions, plotClassProbabilities.
- Dual plotting backend: plotnine (preferred) and matplotlib (fallback).
- Silhouette metric: silhouette_score_long.
- R backend delegation via rpy2 for methods not yet ported.
- Markdown report generation: lcModelReport.
- GitHub Actions CI workflow.
