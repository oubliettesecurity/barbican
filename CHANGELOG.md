# Changelog

All notable changes to `oubliette-barbican` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-10-06

### Fixed
- **PyPI license metadata.** Ship `LICENSE` and `NOTICE` in the sdist and wheel
  and declare Apache-2.0 via PEP 639 (`license = "Apache-2.0"`,
  `license-files`) so the Apache-2.0 relicense from #6 reaches PyPI on the
  next publish. Previously `0.1.0` on PyPI still advertised the pre-relicense
  metadata.

### Added
- This changelog.

## [0.1.0] - 2026-07-26

### Added
- Initial public release of BARBICAN (defensive twin of SPECTRE): coordination
  detection, clustering, embedding helpers, CLI, and demo corpus.
- Zero runtime dependencies (stdlib only); content-scanner signals vendored.
- PyPI Trusted Publishing workflow (`.github/workflows/publish.yml`) and
  packaging-boundary / version-sync tests.

[Unreleased]: https://github.com/oubliettesecurity/barbican/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/oubliettesecurity/barbican/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/oubliettesecurity/barbican/releases/tag/v0.1.0
