# PBS placement

- User instruction (2026-09-21): submit future PBS jobs only to
  `bml2-slave5`, `bml2-slave6`, `bml2-slave7`, or `bml2-slave8`.
- Include an explicit allowed host in each resource request; do not submit
  unconstrained jobs or request slave1–4 unless the user changes this policy.
- Preserve completed outputs and frozen source snapshots when rerunning analyses.
