"""
Data models for the Talk Allocation problem (UCO Metaheuristics Practice 3).

Three core entities:
  - School: institution that requests one or more talks.
  - Talk:   a single talk request from a school (topic + level).
  - Researcher: a researcher who offers to give a talk.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class School:
    """Represents a school that requests scientific talks."""
    school_id: str                  # e.g. "school1"
    location: str                   # "city" | "province"
    disadvantaged_area: bool        # True if in a disadvantaged area
    school_type: str                # "public" | "concerted" | "private"
    first_year: bool                # True if participating for the first time

    @property
    def priority_score(self) -> int:
        """Compute a priority ranking (lower = higher priority) for soft constraints."""
        score = 0
        if self.first_year:
            score += 1          # highest priority
        if self.location == "province":
            score += 2
        if self.disadvantaged_area:
            score += 3
        if self.school_type == "public":
            score += 4
        elif self.school_type == "concerted":
            score += 5
        else:                   # private
            score += 6
        return score


@dataclass
class Talk:
    """Represents a single talk request made by a school."""
    talk_id: int                    # 0-indexed position in talks list
    topic: str                      # required topic ("any" means no restriction)
    level: str                      # required educational level
    school_id: str                  # which school requests this talk


@dataclass
class Researcher:
    """Represents a researcher from the University of Córdoba."""
    researcher_id: str              # e.g. "researcher1"
    topic: str                      # their area of expertise
    level: str                      # their preferred talk level
    can_travel: bool                # False = city only
    first_participation: bool       # True if first year participating
    previous_talk_province: bool    # True if gave a talk in the province last year
    previous_school: Optional[str] = None   # school visited last year (if any)
    max_talks: int = 1              # 1 or 2 (if they agreed to repeat)
