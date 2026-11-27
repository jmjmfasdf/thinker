# PBS placement

- User instruction (2026-09-21; expanded 2026-11-26 to also use slave2,
  then explicitly expanded to slave1, slave3, and slave4): submit future PBS
  jobs only to `bml2-slave1` through `bml2-slave8`.
- Include an explicit allowed host in each resource request; do not submit
  unconstrained jobs or request hosts outside slave1–8 unless the user changes this policy.
- Preserve completed outputs and frozen source snapshots when rerunning analyses.
