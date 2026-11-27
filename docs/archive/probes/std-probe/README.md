# std-probe

Checks the Rust std features Zakuro needs on real Horizon: threads with thread-locals, sync primitives, rayon, time, fs, and big allocations.

```
scripts/horizon-std-setup.sh                         # once: patched libc + std sources in build/
(cd probes/std-probe/rust && ../../../scripts/cargo-nx.sh build --release)
(cd probes/std-probe/c && DEVKITPRO=/opt/devkitpro make)
```

Run `c/std_probe.nro` with kickstand `run_nro`. Each check prints `probe <name>: ok|FAIL ...`. It writes a 1 MB file to `sdmc:/switch/uprisingnx/probe/`.
