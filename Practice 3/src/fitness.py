"""
Modular fitness function for the Talk Allocation problem (minimisation).

Fitness = sum of weighted penalties from hard and soft constraints.
All weights live in a single CONFIG dict so they can be tuned externally.
"""

from typing import Dict, List

from models import School, Talk, Researcher

# ---------------------------------------------------------------------------
# Default weight configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, float] = {
    # Hard constraint penalties
    "w_school_no_talk": 1_000_000,   # school receives 0 talks when R >= T
    "w_location_mismatch": 1_000_000, # researcher cannot physically reach the school

    # Soft constraint penalties (tunable)
    "w_first_year_school": 500,       # school never participated before gets no talk
    "w_province_school": 400,         # province school gets no talk
    "w_disadvantaged": 300,           # disadvantaged area school gets no talk
    "w_public_institution": 200,      # public school gets no talk (lower than above)
    "w_concerted": 100,               # concerted school preference over private

    # Researcher soft priorities (active when R > T)
    "w_first_year_researcher": 200,   # new researcher not used when T < R
    "w_travel_researcher": 150,       # traveller not used to cover province

    # Historical soft penalties
    "w_same_school_as_last_year": 300,  # researcher repeats same school
    "w_same_province_as_last_year": 150, # researcher repeats province after being there last year

    # Overallocation penalty (same researcher assigned more than allowed)
    "w_overallocation": 1_000_000,

    # Direct penalty per unassigned talk (soft, helps differentiate similar solutions)
    "w_unassigned_talk": 10,
}


# ---------------------------------------------------------------------------
# Individual soft-constraint functions
# ---------------------------------------------------------------------------

def _penalty_unserved_school_soft(
    resolved: List[str | None],
    talks: List[Talk],
    schools: Dict[str, School],
    config: Dict[str, float],
) -> float:
    """Soft penalties for schools that receive no talk, weighted by their attributes."""
    assigned_schools = {talks[t_id].school_id for t_id, r_str in enumerate(resolved) if r_str is not None}
    total = 0.0
    for s_id, school in schools.items():
        if s_id in assigned_schools:
            continue
        if school.first_year:
            total += config["w_first_year_school"]
        if school.location == "province":
            total += config["w_province_school"]
        if school.disadvantaged_area:
            total += config["w_disadvantaged"]
        if school.school_type == "public":
            total += config["w_public_institution"]
        elif school.school_type == "concerted":
            total += config["w_concerted"]
    return total


def _penalty_researcher_soft(
    resolved: List[str | None],
    researchers: Dict[str, Researcher],
    num_talks: int,
    config: Dict[str, float],
) -> float:
    """
    Soft penalties active when more researchers are available than talks needed (R > T).
    We penalise when a first-timer or traveller is NOT used (they should be preferred).
    """
    num_researchers = len(researchers)
    if num_researchers <= num_talks:
        return 0.0

    used_ids = {r_str for r_str in resolved if r_str is not None}
    total = 0.0
    for r in researchers.values():
        if r.researcher_id in used_ids:
            continue
        if r.first_participation:
            total += config["w_first_year_researcher"]
        if r.can_travel:
            total += config["w_travel_researcher"]
    return total


def _penalty_historical(
    resolved: List[str | None],
    talks: List[Talk],
    schools: Dict[str, School],
    researcher_index: Dict[str, Researcher],
    config: Dict[str, float],
) -> float:
    """Penalties for historical patterns: same school or same province as last year."""
    total = 0.0
    for talk_id, r_str in enumerate(resolved):
        if r_str is None:
            continue
        r = researcher_index.get(r_str)
        if r is None:
            continue
        school = schools[talks[talk_id].school_id]
        # Same school as last year
        if r.previous_school and r.previous_school == talks[talk_id].school_id:
            total += config["w_same_school_as_last_year"]
        # Province penalty: if researcher was in province last year, prefer city this year
        if r.previous_talk_province and school.location == "province":
            total += config["w_same_province_as_last_year"]
    return total


# ---------------------------------------------------------------------------
# Main fitness function
# ---------------------------------------------------------------------------

def compute_fitness(
    chromosome: List[int],
    talks: List[Talk],
    schools: Dict[str, School],
    researchers: Dict[str, Researcher],
    valid_map: Dict[int, List[str]],
    config: Dict[str, float] | None = None,
    infeasible_schools: set | None = None,
) -> float:
    """
    Compute fitness for a chromosome (minimisation — lower is better).

    Args:
        chromosome: list of length T; chromosome[i] = researcher_id (str key index)
                    or -1 if unassigned.
        talks:      ordered list of Talk objects.
        schools:    dict school_id -> School.
        researchers: dict researcher_id -> Researcher.
        valid_map:  preprocessed dict talk_id -> [valid_researcher_ids].
        config:     penalty weight dictionary (defaults to DEFAULT_CONFIG).

    Returns:
        float fitness (0 = perfect solution, higher = worse).
    """
    if config is None:
        config = DEFAULT_CONFIG

    if infeasible_schools is None:
        infeasible_schools = set()

    num_talks = len(talks)
    num_researchers = len(researchers)

    # Map integer IDs in chromosome back to string keys
    # Chromosome stores indices (0..R-1) or -1; we need the string researcher_id
    researcher_ids = list(researchers.keys())  # ordered index -> str key

    def resolve(idx: int) -> str | None:
        if idx == -1 or idx >= len(researcher_ids):
            return None
        return researcher_ids[idx]

    resolved = [resolve(idx) for idx in chromosome]  # list[str | None]

    fitness = 0.0

    # --- Hard: location mismatch ---
    for talk_id, r_str in enumerate(resolved):
        if r_str is None:
            continue
        r = researchers.get(r_str)
        if r is None:
            continue
        school = schools[talks[talk_id].school_id]
        if school.location == "province" and not r.can_travel:
            fitness += config["w_location_mismatch"]

    # --- Hard: school without talk when R >= T ---
    # Exclude structurally infeasible schools (all their talks have V(t) = ∅)
    if num_researchers >= num_talks:
        assigned_schools = {talks[t_id].school_id for t_id, r_str in enumerate(resolved) if r_str is not None}
        for s_id in schools:
            if s_id in infeasible_schools:
                continue
            if s_id not in assigned_schools:
                fitness += config["w_school_no_talk"]

    # --- Hard: overallocation ---
    counts: Dict[str, int] = {}
    for r_str in resolved:
        if r_str:
            counts[r_str] = counts.get(r_str, 0) + 1
    for r_str, cnt in counts.items():
        r = researchers.get(r_str)
        if r and cnt > r.max_talks:
            fitness += (cnt - r.max_talks) * config["w_overallocation"]

    # --- Soft: unserved school penalties by priority ---
    fitness += _penalty_unserved_school_soft(
        resolved, talks, schools, config,
    )

    # --- Soft: unused researcher penalties (R > T case) ---
    fitness += _penalty_researcher_soft(
        resolved, researchers, num_talks, config,
    )

    # --- Soft: historical penalties ---
    fitness += _penalty_historical(
        resolved, talks, schools, researchers, config,
    )

    # --- Soft: unassigned talk penalty (differentiates similar solutions) ---
    n_unassigned = sum(1 for r_str in resolved if r_str is None)
    fitness += n_unassigned * config.get("w_unassigned_talk", 10)

    return fitness


# ---------------------------------------------------------------------------
# Penalty breakdown for analysis / visualization
# ---------------------------------------------------------------------------

def compute_penalty_breakdown(
    chromosome: List[int],
    talks: List[Talk],
    schools: Dict[str, School],
    researchers: Dict[str, Researcher],
    config: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    Compute individual penalty components for analysis/visualization.

    Returns a dict with:
    - "Unused researchers": penalty for first-year or traveller researchers not used
    - "Historical": penalty for repeating same school or province
    - "Unserved schools (soft)": penalty for schools with no talk
    - "Unassigned talks": penalty for unassigned talks
    """
    if config is None:
        config = DEFAULT_CONFIG

    researcher_ids = list(researchers.keys())

    def resolve(idx: int) -> str | None:
        if idx == -1 or idx >= len(researcher_ids):
            return None
        return researcher_ids[idx]

    resolved = [resolve(idx) for idx in chromosome]
    num_talks = len(talks)
    num_researchers = len(researchers)

    n_unassigned = sum(1 for r_str in resolved if r_str is None)

    unused_penalty = _penalty_researcher_soft(
        resolved, researchers, num_talks, config,
    )

    hist_penalty = _penalty_historical(
        resolved, talks, schools, researchers, config,
    )

    unserved_penalty = _penalty_unserved_school_soft(
        resolved, talks, schools, config,
    )

    unassigned_penalty = n_unassigned * config.get("w_unassigned_talk", 10)

    return {
        "Unused researchers": unused_penalty,
        "Historical": hist_penalty,
        "Unserved schools (soft)": unserved_penalty,
        "Unassigned talks": unassigned_penalty,
    }


# ---------------------------------------------------------------------------
# Helper to compute structurally infeasible schools
# ---------------------------------------------------------------------------

def compute_infeasible_schools(
    talks: List[Talk],
    schools: Dict[str, School],
    valid_map: Dict[int, List[str]],
) -> set:
    """
    Compute the set of schools that are structurally infeasible — i.e.,
    ALL their talks have no valid researchers (V(t) = ∅).

    These schools cannot receive any talk regardless of the assignment,
    so we exclude them from the w_school_no_talk hard penalty.
    """
    school_to_talks: Dict[str, List[int]] = {}
    for t in talks:
        school_to_talks.setdefault(t.school_id, []).append(t.talk_id)

    infeasible: set = set()
    for s_id, talk_ids in school_to_talks.items():
        all_empty = all(len(valid_map.get(tid, [])) == 0 for tid in talk_ids)
        if all_empty:
            infeasible.add(s_id)

    return infeasible


# ---------------------------------------------------------------------------
# Chromosome gene repair helper
# ---------------------------------------------------------------------------

def repair_gene(talk_id: int, valid_map: Dict[int, List[str]], researcher_ids: List[str]) -> int:
    """
    Return a valid researcher index for a given talk, chosen from valid_map.
    If no valid researcher exists, return -1.
    """
    import random
    candidates = valid_map.get(talk_id, [])
    if not candidates:
        return -1
    r_str = random.choice(candidates)
    return researcher_ids.index(r_str)
