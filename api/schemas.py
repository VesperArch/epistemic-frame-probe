"""Pydantic schemas for epistemic frame classification."""

from enum import Enum
from pydantic import BaseModel, Field


class FrameType(str, Enum):
    """Taxonomy of epistemic frames observed in LLM prompt analysis."""

    neutral = "neutral"
    hypothetical = "hypothetical"
    delegated = "delegated"
    authority = "authority"
    emotional = "emotional"
    unknown = "unknown"


class EpistemicMechanism(str, Enum):
    """Cognitive mechanism by which a frame shifts model evaluation behavior."""

    none = "none"
    responsibility_transfer = "responsibility_transfer"
    authority_deactivation = "authority_deactivation"
    narrative_cooptation = "narrative_cooptation"
    academic_redirection = "academic_redirection"


class Recommendation(str, Enum):
    """Action recommendation for the downstream pipeline based on frame risk."""

    pass_input = "pass_input"
    flag = "flag"
    block = "block"


class FrameClassification(BaseModel):
    """Complete classification output for a single input text.

    Attributes:
        input_text: The original text submitted for classification.
        frame_detected: The dominant epistemic frame identified in the input.
        frame_confidence: Classifier confidence in the detected frame, 0.0 to 1.0.
        frame_markers: Lexical or structural markers that triggered frame detection.
        semantic_content: The underlying request with frame markers stripped.
        epistemic_mechanism: The cognitive mechanism the frame activates.
        epistemic_risk_score: Estimated probability the frame bypasses safety evaluation, 0.0 to 1.0.
        risk_factors: Specific elements that contribute to the risk score.
        recommendation: Suggested action for the downstream pipeline.
        justification: Explanation of the classification decision.
        provider_used: LLM provider that performed the classification.
        model_used: Specific model identifier used for classification.
        latency_ms: Time taken for the LLM call in milliseconds.
    """

    input_text: str = Field(..., description="Original text submitted for classification.")
    frame_detected: FrameType = Field(..., description="Dominant epistemic frame.")
    frame_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in detected frame.")
    frame_markers: list[str] = Field(default_factory=list, description="Markers that triggered frame detection.")
    semantic_content: str = Field(..., description="Underlying request with frame markers removed.")
    epistemic_mechanism: EpistemicMechanism = Field(..., description="Cognitive mechanism activated by the frame.")
    epistemic_risk_score: float = Field(..., ge=0.0, le=1.0, description="Estimated bypass probability.")
    risk_factors: list[str] = Field(default_factory=list, description="Elements contributing to risk score.")
    recommendation: Recommendation = Field(..., description="Suggested action for the downstream pipeline.")
    justification: str = Field(..., description="Explanation of the classification decision.")
    provider_used: str = Field(..., description="LLM provider that performed classification.")
    model_used: str = Field(..., description="Specific model identifier used.")
    latency_ms: float = Field(..., ge=0.0, description="LLM call duration in milliseconds.")


class ClassifyRequest(BaseModel):
    """Request body for single-text classification."""

    text: str = Field(..., min_length=1, description="Input text to classify.")


class ClassifyBatchRequest(BaseModel):
    """Request body for batch classification."""

    texts: list[str] = Field(..., min_length=1, description="List of input texts to classify.")


class HealthResponse(BaseModel):
    """Health check response including active provider information."""

    status: str
    provider: str
    model: str


class FrameTaxonomyEntry(BaseModel):
    """Single entry in the frame taxonomy reference."""

    frame: FrameType
    trigger_pattern: str
    mechanism: EpistemicMechanism
    risk_range: str
    description: str
    examples: list[str]


class TaxonomyResponse(BaseModel):
    """Full frame taxonomy with examples."""

    frames: list[FrameTaxonomyEntry]
