# Track 3 Submission — Baseline + Best Systems Diagnosis

> **Locked submission:** the semantics-preserving ABIDES baseline (reproduces all 65 public
> units bit-exact) + the Best Systems Diagnosis `profile.json`. This is an admissible,
> competitive entry. The Rust compiled core is a future speed stretch, not part of this
> submission (it is ~90% to bit-exact; see the work ledger / SPEC-03).

## What is submittable (and already verified)

The submission is a **Docker image** (uploaded as a sealed zip, not an image reference). Ours:

| Requirement (SUBMISSION_CLI.md) | Status |
|---|---|
| `linux/amd64` image | ✅ `track3-abides-baseline:latest` |
| `qfbench2.interface_version="2.0"` label | ✅ verified |
| No Docker `VOLUME` | ✅ verified (`null`) |
| `simulate` + `simulate-batch` verbs on PATH | ✅ verified |
| Reproduces reference traces (admissibility) | ✅ **65/65 public units pass**, bit-exact |
| Best Systems Diagnosis `profile.json` | ✅ `submission/profile.json`, verify_profile quality 1.0 |

The image was built from `baselines/` and passes the full local regression
(`regression_suite/run_regression.py` → 65/65) and the bit-exact equivalence harness.

## The remaining steps — THESE NEED YOUR CREDENTIALS (I can't do them)

An agent cannot push to your registry or hold your Team Key. Here is the exact sequence:

**1. Push the image to an anonymously-pullable registry** (Docker Hub, GHCR public, ECR public):
```bash
docker tag track3-abides-baseline:latest <your-registry>/track3-abides-baseline:latest
docker push <your-registry>/track3-abides-baseline:latest
# capture the pushed digest:
docker inspect --format='{{index .RepoDigests 0}}' <your-registry>/track3-abides-baseline:latest
```
Confirm it pulls anonymously (from a logged-out shell): `docker pull <registry>@sha256:<digest>`.

**2. Fill the descriptor** `submission/submission.json`: put your `<registry>/track3-abides-baseline`
in `image.registry` and the `sha256:...` digest in `image.digest`.

**3. Pack the submission** (prompts for your Team Key on a hidden prompt):
```bash
qfbench2 submission pack --descriptor submission/submission.json \
    --team-number <YOUR_TEAM_NUMBER> --out submission.zip
```
This writes `submission.zip` (contains `submission.json` + `team-claim.json`). The Team Key
never enters the zip.

**4. Upload `submission.zip`** on the Track 3 CodaBench page from your team's designated account.
Dev limit: 5 uploads/team/day, 20 total during Development.

## Notes
- Keep the image anonymously pullable until results publish.
- The `profile.json` here is produced by `profiling/make_simprofile.py` from a real profiled
  run; the baseline image can emit it per-run if you want it bundled alongside each trace.
- Development board closed 2026-10-12; we are in the Final window (→ 2026-10-25).
