# Configured model tiers

The existing local `.env` tier settings use these candidates from the already
verified DigitalOcean `/v1/models` response. No second model-list request or live
completion was made. Runtime routing must read the environment variables rather
than hardcode these IDs.

| Setting | Selected ID | Selection rationale | USD per 1M input / output tokens |
| --- | --- | --- | --- |
| `MODEL_SMALL` | `glm-5.3-flash` | Lowest-priced preferred candidate for text classification and routine replies. Classification does not require tool calls. | $0.15 / $0.50 |
| `MODEL_MID` | `glm-5.3` | Preferred general-purpose candidate for logistics, priced below the large tier. | $1.40 / $4.40 |
| `MODEL_LARGE` | `qwen3.8-max` | Preferred high-capability candidate for direct urgent replies. | $2.00 / $6.00 |

Tool compatibility is handled through the OpenAI-compatible completion interface:
use native tool calls when returned, then the approved JSON-block/plain-reply
fallbacks. Catalog availability does not prove native tool support for an
individual model. No live tool-capability request has been authorized or tested;
these choices must not be advertised as live tool-validated.

Pricing source: https://docs.digitalocean.com/products/inference/details/pricing/
(page updated 2026-10-09, retrieved during mission preparation on 2026-10-12).
These published rates are estimates for cost accounting, not a bill or a claim
that inference has run.
