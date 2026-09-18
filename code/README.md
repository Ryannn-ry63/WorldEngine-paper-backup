# Selector paper research checkout

Independent source snapshot for the next causal experiments. This is a research checkout, not the final public release.

See `docs/20260916_EXECUTION.md` for the current research protocol. Dataset/model paths live in the sibling `registry/`; outputs live in sibling `runs/` and `reports/`. The original workspace remains the immutable artifact source.

The initial snapshot preserves existing implementation dependencies. Legacy experiment implementations are retained until their dependencies can be removed with behavioral tests. New studies use the unified `selector_study` entry point.
