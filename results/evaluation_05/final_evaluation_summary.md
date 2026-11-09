# Comparative Evaluation Results

## Experimental Setup

- Base model: Qwen2.5-1.5B
- Checkpoints: Base Qwen, CPT, SFT, DPO
- Evaluation scenarios: 10 synthetic technician notes
- Evidence preservation: Human-reviewed
- Schema compliance: Automated exact-field matching
- Unsupported claims: Human-reviewed report-level flags

## Results

| stage     |   verified_recall |   schema_compliance |   truncated_reports |   scenarios |   schema_coverage_corrected |   full_schema_compliance |
|:----------|------------------:|--------------------:|--------------------:|------------:|----------------------------:|-------------------------:|
| Base Qwen |             0.943 |               0.1   |                   3 |          10 |                       0.1   |                        0 |
| CPT       |             0.658 |               0.129 |                   1 |          10 |                       0.129 |                        0 |
| DPO       |             0.583 |               1     |                   0 |          10 |                       1     |                        1 |
| SFT       |             0.427 |               1     |                   0 |          10 |                       1     |                        1 |

## Interpretation

This evaluation compares evidence preservation, report standardization,
unsupported claims, and truncation across the four model stages.

The results are descriptive and do not establish industrial reliability.
DPO preference accuracy must be interpreted separately from downstream
report quality.

## Limitations

- Small, synthetic test benchmark.
- Potential semantic similarities between training and test scenarios.
- Single-reviewer judgments without independent adjudication.
- Exact schema compliance does not measure overall report readability.
- The contribution of continued pretraining is not isolated without
  a direct-SFT control experiment.
