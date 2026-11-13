# nothing-else-broke

The five-layer **Nothing-Else-Broke audit** for an intervened language model, scored with Lexsi's AuditKIT
and re-judged with SafeTune. Given a base model and the same model with an intervention applied, it
measures what the intervention did and what else it touched:

| layer | module | measures |
|---|---|---|
| L1 behavioural | `behavioural` | edited facts, neighbourhood and paraphrase accuracy (answer matching with aliases and transliteration variants), MMLU, MGSM, IFEval, WikiText perplexity |
| L2 safety | `behavioural`, `judge` | refusal on held-out harmful, other-category, near-miss benign and XSTest-safe prompts; AuditKIT `RefusalDetector` plus multilingual onsets as the primary judge, SafeTune's refusal detector as the second judge (`refused_safetune`) |
| L3 cross-lingual | `behavioural` | the L1/L2 families in every configured language |
| L4 mechanistic | `mechanistic` | EAP-IG re-attribution under the intervention, cross-circuit CAD on induced edge masks with same-data repeat and resample floors, AS-NC, circuit distance |
| L5 hidden failures | `behavioural.eval_hidden` | never-edited facts: margin spread, AURC over confidence, high-confidence error |

Every family returns an AuditKIT `RunResult` (`ak.Sample` / `ak.scorer` / `ak.evaluate`), so the
before/after comparison is AuditKIT's paired bootstrap on the same sample ids. Package `neb`; depends on
[tiny-aya-circuits](../tiny-aya-circuits) for model loading, the chat format and circuit tasks. The study
that drives it (cells, hypotheses, report) is [circuit-steering](../circuit-steering): its
`audit/run_audit.py` applies each cell and calls these layers, its `audit/analyze.py` turns the
AuditKIT runs into the pre-registered tests.

```bash
uv sync && uv run pytest -q
```
