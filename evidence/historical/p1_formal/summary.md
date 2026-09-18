# P1 balance / formal: PASS

Separate development/native protocols; all outcomes retained.

Decision: REVIEW_REQUIRED. No automatic promotion or next experiment.

| Protocol | Block | Arm | PDM mean | Seed std |
|---|---|---|---:|---:|
| formal | open_navtest | P1 | 0.861561 | 0.006632 |
| formal | open_navtest | CL_NLL | 0.858602 | 0.000208 |
| formal | open_navtest | scalar_v3 | 0.875426 | fixed |
| formal | open_rare | P1 | 0.682308 | 0.002082 |
| formal | open_rare | CL_NLL | 0.672539 | 0.018671 |
| formal | open_rare | scalar_v3 | 0.680170 | fixed |
| formal | full_NR | P1 | 0.790841 | 0.005727 |
| formal | full_NR | CL_NLL | 0.782610 | 0.006504 |
| formal | full_NR | scalar_v3 | 0.776398 | fixed |
| formal | full_R | P1 | 0.784375 | 0.015987 |
| formal | full_R | CL_NLL | 0.779239 | 0.008562 |
| formal | full_R | scalar_v3 | 0.767742 | fixed |
| formal | common_NR | P1 | 0.896172 | 0.019654 |
| formal | common_NR | CL_NLL | 0.892208 | 0.005646 |
| formal | common_NR | scalar_v3 | 0.853193 | fixed |
| formal | common_R | P1 | 0.888383 | 0.025209 |
| formal | common_R | CL_NLL | 0.877541 | 0.011188 |
| formal | common_R | scalar_v3 | 0.876981 | fixed |

Common64 is a retention cohort, not full-common or an unseen test.
Log intervals condition on one acquisition and evaluation noise; they do not replace seed uncertainty.
Three optimization seeds; no observed-best selection. See JSON for all safety components and rescued/broken.

Missing: 0
