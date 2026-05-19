"""
Instance generator for Practice 3 — wraps the professor's
data_generator_talks11f logic and returns School/Talk/Researcher model objects.

Usage:
    from generate_instance import generate_instance
    schools, talks, researchers = generate_instance(seed=42)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow import from the sibling data_generator_talks11f package
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generator_talks11f.schools_functions import generate_random_school
from data_generator_talks11f.talks_functions import (
    generate_random_topic,
    generate_random_talk_level,
)
from data_generator_talks11f.proposed_talks_functions import (
    generate_random_repeat_talk,
    generate_random_travelling,
    generate_random_first_participation,
    generate_random_previous_talk_province,
)

from models import School, Talk, Researcher


def generate_instance(
    num_schools: int = 10,
    num_talks: int = 20,
    num_researchers: int = 15,
    max_talks_researchers: int = 2,
    prob_topics: float = 0.2,
    seed: int | None = None,
    balance_levels: bool = True,
) -> Tuple[Dict[str, School], List[Talk], Dict[str, Researcher]]:
    """
    Generate a random instance using the same logic as the professor's
    data_generator_talks11f.

    Parameters
    ----------
    num_schools : int
    num_talks : int
    num_researchers : int
    max_talks_researchers : int
        Maximum talks a single researcher can give (1 or 2).
    prob_topics : float
        Probability that a talk requests a specific topic (0 = all "any").
    seed : int or None
        Numpy random seed for reproducibility.

    Returns
    -------
    schools      : dict school_id -> School
    talks        : list of Talk
    researchers  : dict researcher_id -> Researcher
    """
    if seed is not None:
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Schools
    # ------------------------------------------------------------------
    schools: Dict[str, School] = {}
    school_ids: List[str] = []
    for i in range(num_schools):
        s = generate_random_school()
        sid = f"school{i + 1}"
        schools[sid] = School(
            school_id=sid,
            location=s["location"],
            disadvantaged_area=s["disadvantaged_area"],
            school_type=s["school_type"],
            first_year=s["first_year"],
        )
        school_ids.append(sid)

    # ------------------------------------------------------------------
    # Requested talks — first assign one per school, then fill randomly
    # ------------------------------------------------------------------
    talks: List[Talk] = []
    seen: set = set()

    # First: guarantee at least one talk per school
    for sid in school_ids:
        topic = generate_random_topic() if np.random.rand() < prob_topics else "any"
        level = generate_random_talk_level()
        talks.append(Talk(talk_id=len(talks), topic=topic,
                          level=level, school_id=sid))
        seen.add((sid, topic, level))

    # Then fill remaining slots avoiding duplicates
    max_attempts = num_talks * 10
    attempts = 0
    while len(talks) < num_talks and attempts < max_attempts:
        sid = np.random.choice(school_ids)
        topic = generate_random_topic() if np.random.rand() < prob_topics else "any"
        level = generate_random_talk_level()
        key = (sid, topic, level)
        if key not in seen:
            seen.add(key)
            talks.append(Talk(talk_id=len(talks), topic=topic,
                              level=level, school_id=sid))
        attempts += 1

    # Fallback: fill remaining slots without duplicate filtering
    while len(talks) < num_talks:
        sid = np.random.choice(school_ids)
        topic = generate_random_topic() if np.random.rand() < prob_topics else "any"
        level = generate_random_talk_level()
        talks.append(Talk(talk_id=len(talks), topic=topic,
                          level=level, school_id=sid))

    # ------------------------------------------------------------------
    # Researchers
    # ------------------------------------------------------------------
    province_schools = [s for s in school_ids
                        if schools[s].location == "province"]
    city_schools = [s for s in school_ids
                    if schools[s].location == "city"]

    researchers: Dict[str, Researcher] = {}

    for i in range(num_researchers):
        repeat = generate_random_repeat_talk()
        can_travel = generate_random_travelling()
        first_part = generate_random_first_participation()
        prev_prov = generate_random_previous_talk_province()
        topic = generate_random_topic()
        level = generate_random_talk_level()

        # Determine previous school (consistent with location history)
        if prev_prov and province_schools:
            prev_school = str(np.random.choice(province_schools))
        elif city_schools:
            prev_school = str(np.random.choice(city_schools))
            prev_prov = False
        else:
            prev_school = str(np.random.choice(school_ids))
            prev_prov = False

        max_t = max_talks_researchers if repeat else 1

        researchers[f"researcher{i + 1}"] = Researcher(
            researcher_id=f"researcher{i + 1}",
            topic=topic,
            level=level,
            can_travel=can_travel,
            first_participation=first_part,
            previous_talk_province=prev_prov,
            previous_school=prev_school,
            max_talks=max_t,
        )

    # Balance researcher levels so every level has at least the minimum
    # needed to cover its fair share of talks (avoids infeasible seeds)
    if balance_levels:
        from collections import Counter
        all_levels = ["preschool", "primary", "secondary",
                      "high school", "vocational training"]
        level_counts = Counter(r.level for r in researchers.values())
        min_per_level = max(2, (num_talks // len(all_levels) + 1) // 2)

        for lvl in all_levels:
            while level_counts.get(lvl, 0) < min_per_level:
                donor = next((r for r in researchers.values()
                             if level_counts.get(r.level, 0) > min_per_level), None)
                if donor is None:
                    break
                level_counts[donor.level] -= 1
                donor.level = lvl
                level_counts[lvl] = level_counts.get(lvl, 0) + 1

    return schools, talks, researchers
