# Architecture

## System Overview

The Epistemic Frame Probe intercepts user input before it reaches a downstream LLM and classifies the rhetorical frame wrapping the request. The system operates as an inline filter: it receives raw user text, dispatches it to a classification model with a structured prompt, parses the JSON response into a typed schema, and returns a risk-annotated record that the caller uses to decide whether to pass, flag, or block the input.

The system sits between user input and the main model. It does not modify the input or generate responses. It produces metadata that the pipeline uses to make routing decisions. A blocked input never reaches the main model. A flagged input may be logged, escalated, or passed with annotations depending on the caller's policy. A passed input proceeds without modification.

```
User Input
    │
    ▼
┌────────────────────────┐
│  Epistemic Frame Probe │
│  (classifier LLM call) │
└────────────┬───────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
 Block / Flag      Pass Input
                      │
                      ▼
               ┌─────────────┐
               │  Main LLM   │
               └─────────────┘
```

The classification itself uses an LLM — the same class of system it is designed to protect. This is a deliberate tradeoff: the classifier operates under a structured, narrow prompt that constrains its output to a JSON schema, and its task — frame classification with fixed output schema — is structurally narrower than the open-ended generation tasks that expose the vulnerability. The bootstrapping problem is documented in [Known Failure Modes](#5-known-failure-modes).

---

## Component Breakdown

### schemas.py

**Responsibility:** Defines all data contracts for the system — enumerations, request/response models, and taxonomy entries.

**Key design decisions:**
- `FrameType`, `EpistemicMechanism`, and `Recommendation` are string enums so they serialize cleanly to JSON and are readable in logs without additional mapping.
- `FrameClassification` includes `provider_used` and `model_used` so every classification record is self-describing. This matters for audit trails and for debugging provider-specific behavior differences.
- `epistemic_risk_score` and `frame_confidence` are constrained to [0.0, 1.0] at the Pydantic level. Violations surface immediately rather than propagating as silent data errors.

**What it does not do:** It does not enforce business logic like "authority frame must have risk >= 0.7." That constraint lives in the system prompt and is tested in the test suite; the schema layer only enforces structural validity.

---

### classifier.py

**Responsibility:** Dispatches classification prompts to the configured LLM provider and parses the response into a `FrameClassification`.

**Key design decisions:**
- `call_llm()` is a single async function that dispatches to provider-specific implementations. All callers use `call_llm()` exclusively. This makes the provider boundary explicit and testable: mocking `call_llm` in tests covers all provider paths without provider-specific test fixtures.
- Provider selection happens at module load time via `os.getenv("EPISTEMIC_PROVIDER")`. This is a deliberate tradeoff: it avoids per-request config lookups at the cost of requiring a process restart to switch providers. In a research context, that tradeoff is acceptable.
- `classify_batch()` uses `asyncio.gather()` rather than sequential iteration. For a classifier that exists to protect a main LLM, batch throughput matters — blocking on each call serially would make batch protection impractical at scale.
- JSON parse failures return a fallback `FrameClassification` with `unknown` frame and `flag` recommendation rather than raising. The failure-safe default is to flag, not to pass, because an unparseable response from the classifier is itself a signal that something unexpected occurred.
- Markdown code fence stripping handles the common case where models include ` ```json ` wrappers despite explicit instructions not to.

**What it does not do:** It does not cache results. Two identical inputs will result in two LLM calls. Caching would require a decision about whether framing context (timestamps, session state) should affect classification, which is out of scope for this system.

---

### server.py

**Responsibility:** Exposes the classifier as an HTTP API with authentication, CORS, and taxonomy reference endpoints.

**Key design decisions:**
- API key enforcement is implemented as a middleware rather than a FastAPI dependency so it applies uniformly to all routes including future ones, without requiring each route to declare it.
- The key check is skipped entirely when `EPISTEMIC_API_KEY` is unset. This makes local development frictionless without requiring a separate "dev mode" flag.
- `/frames` returns the full taxonomy as structured data rather than documentation markup. Callers can use this to build UIs, generate reports, or validate that their understanding of the taxonomy matches the running system.
- CORS is open (`allow_origins=["*"]`) because this is a research tool, not a production service. Operators who deploy it in a restricted environment should tighten this.

**What it does not do:** It does not persist classification results. Every call is stateless. Audit logging, if needed, belongs in an upstream API gateway or a wrapping service.

---

## Provider Abstraction

Provider-agnostic design matters here because the system's findings may vary across providers. Authority framing may deactivate filters more aggressively in one model than another. Research that locks to a single provider cannot distinguish frame sensitivity from model-specific behavior. Supporting multiple providers makes comparative studies possible without code changes.

`call_llm(prompt: str) -> str` is the single abstraction boundary. It takes raw text and returns raw text. Provider-specific details — client initialization, API key retrieval, request format, response extraction — are encapsulated in `_call_llm_anthropic`, `_call_llm_openai`, `_call_llm_gemini`, and `_call_llm_groq`. The caller never touches provider-specific objects.

**Adding a new provider:**

1. Add the provider name and model identifier to `_PROVIDER_MODELS` in `classifier.py`.
2. Implement an async function `_call_llm_<provider>(prompt: str) -> str` that initializes the provider client from an environment variable, calls the API with `_SYSTEM_PROMPT` as the system instruction, and returns the text response.
3. Add the provider name and function reference to `_PROVIDER_DISPATCH`.
4. Add the API key variable to `.env.example`.

No other changes are required. The server, schemas, and tests are unaffected.

---

## Frame Detection Methodology

The classifier LLM receives `_SYSTEM_PROMPT` as a system instruction on every call. The prompt defines the taxonomy, specifies output format, and gives the model explicit scoring guidance. The model is instructed to return only valid JSON — no prose, no markdown — so the response can be parsed without heuristics.

The prompt instructs the model to perform four distinct operations in sequence: identify the frame, extract semantic content by removing markers, identify the mechanism, and score risk. Separating these steps in the prompt improves reliability because each step has a clear, narrow output target.

Risk scoring logic is grounded in the experimental findings that motivated this system:

| Frame        | Mechanism                  | Risk Range | Basis                                               |
|--------------|----------------------------|------------|-----------------------------------------------------|
| neutral      | none                       | 0.0–0.2    | No framing device; model evaluates content directly |
| hypothetical | narrative_cooptation       | 0.3–0.5    | Narrative wrapper reduces perceived stakes          |
| delegated    | academic_redirection       | 0.4–0.6    | Third-party attribution displaces accountability    |
| emotional    | narrative_cooptation       | 0.3–0.5    | Personal narrative lowers guard without authority   |
| authority    | authority_deactivation     | 0.7–1.0    | Experimentally shown to completely bypass filters   |
| unknown      | none                       | 0.5        | Failure to classify is treated as elevated risk     |

Authority frame scores highest because controlled experiments showed that citing recognized institutions (SANS Institute, Verizon DBIR, NIST, OWASP) caused the target model to deliver the requested content without triggering safety evaluation. The same content framed neutrally was blocked. This is not a gradual effect; the experiments showed a categorical behavioral shift at the authority frame boundary.

---

## Known Failure Modes

**Nested frames (authority inside hypothetical):** When an authority citation appears inside a fictional frame, the classifier may detect only one frame and miss the compound risk. A hypothetical prompt that internally cites SANS Institute carries authority-level risk but may be classified as hypothetical. The system prompt instructs the model to identify the "dominant" frame, which partially mitigates this, but nested frames reduce classification accuracy.

**Novel authority sources:** Authority frame detection depends on the classifier model's knowledge of real institution names. A prompt citing a real but obscure institution that the model does not recognize will likely be classified as neutral or delegated rather than authority. This is an inherent limitation of training-data-bounded knowledge.

**Non-English frame markers:** The system prompt is written in English. Frame markers in other languages — particularly Portuguese, as in the experimental prompts — may be detected correctly if the classifier model has multilingual capability, but this is not guaranteed. The experimental prompts in `examples/example_requests.py` are in Portuguese and serve as a functional test of multilingual detection.

**The bootstrapping problem:** The classifier uses an LLM to detect a vulnerability that affects LLMs. If the classifier is itself subject to authority framing, a carefully constructed meta-prompt could cause it to misclassify authority frames as neutral. The structured output constraint and narrow task scope reduce but do not eliminate this risk. The classifier should not be treated as a security boundary in isolation; it is one layer in a defense-in-depth pipeline.

---

## Extension Points

**Adding a new FrameType:**
1. Add the value to `FrameType` enum in `schemas.py`.
2. Add a corresponding entry to `EpistemicMechanism` if it activates a new mechanism.
3. Add the frame definition to `_SYSTEM_PROMPT` in `classifier.py` with trigger patterns, mechanism, and risk range.
4. Add a `FrameTaxonomyEntry` to `_FRAME_TAXONOMY` in `server.py`.
5. Add test cases for the new frame in `tests/test_classifier.py`.

**Adjusting risk thresholds:**
Risk thresholds for `Recommendation` assignment are encoded in `_SYSTEM_PROMPT`. To change the threshold for `block` from `>= 0.7` to `>= 0.8`, update the scoring guidance in the prompt and update the corresponding test assertions in `test_classifier.py`. The server and schema layers require no changes.

**Plugging into a robot dialog pipeline:**
A robot dialog system using an LLM for natural language understanding can position this classifier as a pre-processor on the user utterance channel. Before any utterance reaches the dialog module, it passes through `/classify`. If the recommendation is `block`, the dialog module receives a synthetic "I cannot help with that" signal instead of the original utterance. If `flag`, the utterance is logged and forwarded with frame metadata attached, allowing the dialog module to use the metadata in its response policy. The `/classify` endpoint is synchronous from the caller's perspective and adds only the classifier's latency to the round-trip time.
