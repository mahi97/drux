# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Shared numerical core (`drux.numerical`) for time-grid construction and array-native evaluation
- Public `evaluate(t)` method on every model (scalar or NumPy array)
- Shared test suite covering all five models (fractional steps, non-finite input, rollback, repeated runs, formula equivalence)
### Changed
- Model equations now use vectorized NumPy operations instead of `np.vectorize` over scalar `math` functions
- `simulate()` timeline always starts at 0 and finishes at the exact requested duration
### Fixed
- Time array could contain a point after `duration` when the duration was not divisible by the time step
- A failed `simulate()` call could overwrite `_time_points` before raising, leaving a partial result
## [0.4] - 2026-05-18
### Added
- Hopfenberg model
## [0.3] - 2025-12-08
### Added
- Weibull model
- Logo
- `__repr__` method for models
### Changed
- `Python 3.14` added to `test.yml`
## [0.2] - 2025-09-27
### Added
- Zero-order model
- First-order model
### Changed
- `README.md` modified
- Test system modified
- Release range bug in `time_for_release` fixed
- `matplotlib` import bug fixed
- `plot` method modified
## [0.1] - 2025-07-27
### Added
- Base model
- Higuchi model


[Unreleased]: https://github.com/openscilab/drux/compare/v0.4...dev
[0.4]: https://github.com/openscilab/drux/compare/v0.3...v0.4
[0.3]: https://github.com/openscilab/drux/compare/v0.2...v0.3
[0.2]: https://github.com/openscilab/drux/compare/v0.1...v0.2
[0.1]: https://github.com/openscilab/drux/compare/48548f0...v0.1