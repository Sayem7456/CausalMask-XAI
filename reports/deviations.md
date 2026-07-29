# Deviations Log

Record any scientific deviations from the registered plan here.

| Date | Phase | Deviation | Justification | Impact |
|------|-------|-----------|---------------|--------|
| 2026-07-29 | Phase 8 | Background swap disabled during training (swapped=None) per batch counterfactual generation. Swap consistency is monitored during validation only. | In-batch donor selection (partition-aware) during backpropagation adds complexity. Validation monitoring still captures swap divergence. Deferred full swap integration to Phase 9. | Training omits background-consistency gradient signal; pilot loss effectively CE + sufficiency + necessity. Full five-fold training will include swap loss. |
