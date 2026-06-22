# Locked-threshold external decision validation - 2026-06-22

## Protocol

Thresholds are selected only on the Nisqually leave-one-family validation domain and then applied unchanged to SODO/UW and Canterbury.

## FN:FP = 10 result

Canterbury best locked rule: `M1_nonstationary_groundwater_only` at threshold `0.075` with FN `9` and FP `7535`.
Available-case external best rule: `M1_nonstationary_groundwater_only` at threshold `0.075` with external FN `9` and FP `7535` across `1` external domain(s).
Full-domain external best rule: `M0_static_stationary` at threshold `0.325` with external FN `99` and FP `6765` across `2` external domains.

## Claim boundary

Allowed: a locked decision-rule validation: thresholds are selected without Canterbury or SODO/UW retuning, then transferred externally. At FN:FP=10, M1 strongly reduces Canterbury false negatives relative to M0 and has the best available-case external loss, while the full-domain comparison remains restricted to models available in both SODO/UW and Canterbury.

Blocked: do not state that non-stationary models are universally superior; the locked-rule evidence supports domain-conditional false-negative management, not global calibration dominance.
