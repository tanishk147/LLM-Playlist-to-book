"""Grounding / external verification."""
from .grounded_check import VerificationResult, sentence_grounded
from .canonical_refs import CanonicalRefIndex, ExternalConfirmation

__all__ = [
    "VerificationResult",
    "sentence_grounded",
    "CanonicalRefIndex",
    "ExternalConfirmation",
]
