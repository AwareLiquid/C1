# Vendored M1 code — C1 independence record

C1 is an **independent repository**. To run its experiments on any machine
(training server, CI, a fresh clone) it carries its own copy of the M1 code it
needs. It no longer depends on a sibling `../M1` checkout, and the old
`experiments/_m1_bootstrap.py` auto-clone has been removed.

## Provenance

| Field | Value |
|---|---|
| Source repo | https://github.com/AwareLiquid/M1 (public mirror) |
| Commit | `4645e91a62a20ecf26651a6fbe953eab25212ad3` |
| Commit date | 2026-09-11 |
| Vendored by | C1 independence pass (2026-09-12) |

## What was vendored

| Path | Origin | Why |
|---|---|---|
| `mt_lnn/` | `mt_lnn/` @ `4645e91` | the liquid model (`MTLNNConfig`, `MTLNNModel`, layers) |
| `benchmarks/reasoning_tasks.py` | `benchmarks/reasoning_tasks.py` @ `4645e91` | E1 parity task (`gen_parity`, `vocab_size`) |
| `benchmarks/__init__.py` | minimal empty package marker | lets `from benchmarks.reasoning_tasks import ...` resolve without pulling M1's `selective_copy` |

`mt_lnn/` and `benchmarks/reasoning_tasks.py` were verified byte-identical
between the mirror commit `4645e91` and the local `E:\M1` working tree at the
time of vendoring (empty `git diff`).

## Updating

Re-vendor from a newer mirror commit:

```bash
git clone --depth 1 https://github.com/AwareLiquid/M1.git /tmp/M1
# replace mt_lnn/ and benchmarks/reasoning_tasks.py, then update the SHA above
```

Keep the vendored copy in lockstep with the experiments it serves; do not edit
files under `mt_lnn/` here — fix upstream in M1 and re-vendor.
