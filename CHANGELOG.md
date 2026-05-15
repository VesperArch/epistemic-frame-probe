# Changelog

## [0.1.0] — 2026-05-14

### Added
- Frame classifier with support for Anthropic, OpenAI, Gemini, and Groq
- Five frame types: neutral, hypothetical, delegated, authority, emotional
- FastAPI server with /classify, /classify/batch, /health, /frames endpoints
- Experimental results documenting authority frame risk gradient (0.10 → 0.88)
- Bootstrapping failure documented and mitigated in system prompt design
- Pydantic validator for mechanism/frame consistency
- pytest suite — 15 tests, 0 warnings
