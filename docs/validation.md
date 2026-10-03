# Validation of the source release

Validated on 2026-10-03 with Python 3.11.15, JAX 0.5.3 and the pinned dependency versions. These are software checks, not policy-quality or hardware-performance results.

| Check | Result / scope |
| --- | --- |
| Default `pytest` suite | **63 passed**; includes synthetic raw-data conversion through real LeRobot loading and normalization, transform/provenance checks, checkpoint utilities, and loopback WebSocket/health checks |
| Eight-device integration test | **Passed** with eight virtual CPU devices; exercises actual JAX FSDP sharding, validation, Orbax save/resume and parameter restoration using a small synthetic model |
| CLI help | Train, normalization, conversion, local/remote inference, asset prefetch and checkpoint tools all parse successfully |
| Ruff | Source lint and configured formatting checks pass |
| Lockfile | `uv lock --check --offline` passes; the original resolved package versions were retained |
| Packaging | Both the core and client wheels build successfully |
| Source audit | No findings from the included heuristic scanner; manual review is still required |

The test environment reused an existing compatible dependency installation. The lockfile and wheel builds were checked, but a complete dependency installation on a new GPU machine still needs to be verified.

No NVIDIA device was exposed in the validation environment. **A genuine eight-GPU run, full π₀.₅ checkpoint inference, throughput/memory measurements, convergence, and real-robot success have not been validated here.** Virtual CPU tests do not replace those checks. No private weights or data were used in the tests.

The pinned dependency stack emits deprecation warnings and an Orbax restore-sharding warning in the synthetic resume test. These were not test failures; the test resumes on the same topology. Changing training topology across resume has not been validated.

Before claiming a model is ready for deployment or the “best” checkpoint, complete the remaining items in [the release checklist](release_checklist.md).
