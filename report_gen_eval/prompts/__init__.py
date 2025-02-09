"""Prompts for report evaluation."""

from .check_relevance import (
    SYSTEM_PROMPT as CHECK_RELEVANCE_SYSTEM,
    USER_PROMPT as CHECK_RELEVANCE_USER,
)
from .check_negative import (
    SYSTEM_PROMPT as CHECK_NEGATIVE_SYSTEM,
    USER_PROMPT as CHECK_NEGATIVE_USER,
)
from .requires_citation import (
    SYSTEM_PROMPT as REQUIRES_CITATION_SYSTEM,
    USER_PROMPT as REQUIRES_CITATION_USER,
    USER_PROMPT_SHORT as REQUIRES_CITATION_USER_SHORT,
)
from .first_instance import (
    SYSTEM_PROMPT as FIRST_INSTANCE_SYSTEM,
    USER_PROMPT as FIRST_INSTANCE_USER,
)
from .nugget_agreement import (
    SYSTEM_PROMPT as NUGGET_AGREEMENT_SYSTEM,
    USER_PROMPT as NUGGET_AGREEMENT_USER,
)

__all__ = [
    "CHECK_RELEVANCE_SYSTEM",
    "CHECK_RELEVANCE_USER",
    "CHECK_NEGATIVE_SYSTEM",
    "CHECK_NEGATIVE_USER",
    "REQUIRES_CITATION_SYSTEM",
    "REQUIRES_CITATION_USER",
    "REQUIRES_CITATION_USER_SHORT",
    "FIRST_INSTANCE_SYSTEM",
    "FIRST_INSTANCE_USER",
    "NUGGET_AGREEMENT_SYSTEM",
    "NUGGET_AGREEMENT_USER",
]
