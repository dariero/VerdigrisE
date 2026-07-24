"""Executable alchemical-blossom grimoire fixture for VerdigrisE.

Each entry owns one inspectable retrieval role. Three dosage sources create a
same-subject attribution trap, moonpetal and moonflower create a near-synonym
collision, two orchids create a qualifier-sensitive harvest collision, and an
unanswered shelf-life question owns exact abstention behaviour.
"""

from typing import TypedDict

from config import ABSTENTION_PHRASE


class CorpusEntry(TypedDict):
    id: str
    text: str
    grimoire_id: str | None
    folio: int | str | None
    subject: str
    fact_type: str
    condition: str


class GoldenCaseFixture(TypedDict):
    case_id: str
    question: str
    expected_retrieved_id: str | None
    expected_ranked_ids: list[str]
    collision_sibling_ids: list[str]
    expected_value: str | None
    forbidden_values: list[str]
    expected_grimoire_id: str | None
    required_qualifiers: list[str]
    must_contain: list[str]
    forbidden: list[str]
    expected_answer: str
    expect_abstention: bool


# Additional metadata is retrieval- and evaluation-bearing:
#   subject separates near-synonyms and keeps same-subject source conflicts visible;
#   fact_type identifies the collision family;
#   condition carries the qualifier that prevents value-only answers from passing.
CORPUS: list[CorpusEntry] = [
    {
        "id": "moonpetal-silver-vapor",
        "text": (
            "The Nocturne Herbarium states that moonpetal calyxes ground with "
            "11 grains of pearl salt release a silver sleep vapor."
        ),
        "grimoire_id": "GRIM-NOCTURNE",
        "folio": "IV-a",
        "subject": "moonpetal",
        "fact_type": "ground-blossom vapor",
        "condition": "ground with pearl salt",
    },
    {
        "id": "moonflower-golden-vapor",
        "text": (
            "The Dawn Petal Grimoire states that moonflower petals ground with "
            "17 grains of pearl salt release a golden waking vapor."
        ),
        "grimoire_id": "GRIM-DAWN-PETAL",
        "folio": "IV-b",
        "subject": "moonflower",
        "fact_type": "ground-blossom vapor",
        "condition": "ground with pearl salt",
    },
    {
        "id": "verdigris-dose-verdant",
        "text": (
            "For tarnish fever, the Verdant Crucible prescribes a dose of 3 drams "
            "of verdigris blossom elixir when it was distilled in copper and "
            "administered after dusk."
        ),
        "grimoire_id": "GRIM-VERDANT",
        "folio": 21,
        "subject": "verdigris blossom elixir",
        "fact_type": "tarnish-fever dosage",
        "condition": "distilled in copper and administered after dusk",
    },
    {
        "id": "verdigris-dose-amber",
        "text": (
            "For tarnish fever, the Amber Alembic prescribes a dose of 9 drams "
            "of verdigris blossom elixir when it was distilled in amber glass and "
            "administered at dawn."
        ),
        "grimoire_id": "GRIM-AMBER",
        "folio": 22,
        "subject": "verdigris blossom elixir",
        "fact_type": "tarnish-fever dosage",
        "condition": "distilled in amber glass and administered at dawn",
    },
    {
        "id": "verdigris-dose-obsidian",
        "text": (
            "For tarnish fever, the Obsidian Petal Index prescribes a dose of 15 drams "
            "of verdigris blossom elixir when it was distilled in basalt during a "
            "lunar eclipse."
        ),
        "grimoire_id": "GRIM-OBSIDIAN-PETAL",
        "folio": 23,
        "subject": "verdigris blossom elixir",
        "fact_type": "tarnish-fever dosage",
        "condition": "distilled in basalt during a lunar eclipse",
    },
    {
        "id": "shadeglass-orchid-harvest",
        "text": (
            "The Umbral Botany Grimoire permits harvesting a shadeglass orchid after "
            "7 moon-phases only when the plant was grown entirely in shade. Direct sun "
            "exposure invalidates that harvest window."
        ),
        "grimoire_id": "GRIM-UMBRAL-BOTANY",
        "folio": "VII",
        "subject": "shadeglass orchid",
        "fact_type": "harvest interval",
        "condition": "grown entirely in shade",
    },
    {
        "id": "sunspire-orchid-harvest",
        "text": (
            "The Solar Flora Ledger permits harvesting a sunspire orchid after "
            "10 moon-phases only when the plant was grown in full sun."
        ),
        "grimoire_id": "GRIM-SOLAR-FLORA",
        "folio": "VIII",
        "subject": "sunspire orchid",
        "fact_type": "harvest interval",
        "condition": "grown in full sun",
    },
    {
        "id": "asterquartz-powdering",
        "text": (
            "The Lapidary Bloom Codex permits asterquartz to be powdered for elixirs "
            "only at Mohs hardness 8; softer specimens are reserved for lenswork."
        ),
        "grimoire_id": "GRIM-LAPIDARY-BLOOM",
        "folio": 31,
        "subject": "asterquartz",
        "fact_type": "safe powdering hardness",
        "condition": "powdered for elixirs",
    },
]


GOLDEN_CASES: list[GoldenCaseFixture] = [
    {
        "case_id": "numeric-source-verdigris-dose",
        "question": (
            "According to GRIM-VERDANT, what dose of verdigris blossom elixir is "
            "prescribed for tarnish fever when it was distilled in copper and "
            "administered after dusk?"
        ),
        "expected_retrieved_id": "verdigris-dose-verdant",
        "expected_ranked_ids": ["verdigris-dose-verdant", "verdigris-dose-amber"],
        "collision_sibling_ids": ["verdigris-dose-amber"],
        "expected_value": "3 drams",
        "forbidden_values": ["9 drams"],
        "expected_grimoire_id": "GRIM-VERDANT",
        "required_qualifiers": ["distilled in copper", "administered after dusk"],
        "must_contain": [
            "3 drams",
            "distilled in copper",
            "administered after dusk",
            "GRIM-VERDANT",
            "[verdigris-dose-verdant]",
        ],
        "forbidden": [
            "9 drams",
            "GRIM-AMBER",
            "distilled in amber glass",
            "administered at dawn",
            "[verdigris-dose-amber]",
            "15 drams",
            "GRIM-OBSIDIAN-PETAL",
            "distilled in basalt",
            "lunar eclipse",
            "[verdigris-dose-obsidian]",
        ],
        "expected_answer": (
            "GRIM-VERDANT prescribes 3 drams of verdigris blossom elixir for "
            "tarnish fever when distilled in copper and administered after dusk "
            "[verdigris-dose-verdant]."
        ),
        "expect_abstention": False,
    },
    {
        "case_id": "numeric-source-amber-dose",
        "question": (
            "According to GRIM-AMBER, what dose of verdigris blossom elixir is "
            "prescribed for tarnish fever when it was distilled in amber glass and "
            "administered at dawn?"
        ),
        "expected_retrieved_id": "verdigris-dose-amber",
        "expected_ranked_ids": ["verdigris-dose-amber", "verdigris-dose-obsidian"],
        "collision_sibling_ids": ["verdigris-dose-obsidian"],
        "expected_value": "9 drams",
        "forbidden_values": ["15 drams"],
        "expected_grimoire_id": "GRIM-AMBER",
        "required_qualifiers": ["distilled in amber glass", "administered at dawn"],
        "must_contain": [
            "9 drams",
            "distilled in amber glass",
            "administered at dawn",
            "GRIM-AMBER",
            "[verdigris-dose-amber]",
        ],
        "forbidden": [
            "15 drams",
            "GRIM-OBSIDIAN-PETAL",
            "distilled in basalt",
            "lunar eclipse",
            "[verdigris-dose-obsidian]",
            "3 drams",
            "GRIM-VERDANT",
            "distilled in copper",
            "administered after dusk",
            "[verdigris-dose-verdant]",
        ],
        "expected_answer": (
            "GRIM-AMBER prescribes 9 drams of verdigris blossom elixir for tarnish "
            "fever when distilled in amber glass and administered at dawn "
            "[verdigris-dose-amber]."
        ),
        "expect_abstention": False,
    },
    {
        "case_id": "numeric-source-obsidian-dose",
        "question": (
            "According to GRIM-OBSIDIAN-PETAL, what dose of verdigris blossom elixir "
            "is prescribed for tarnish fever when it was distilled in basalt during "
            "a lunar eclipse?"
        ),
        "expected_retrieved_id": "verdigris-dose-obsidian",
        "expected_ranked_ids": ["verdigris-dose-obsidian", "verdigris-dose-verdant"],
        "collision_sibling_ids": ["verdigris-dose-verdant"],
        "expected_value": "15 drams",
        "forbidden_values": ["3 drams"],
        "expected_grimoire_id": "GRIM-OBSIDIAN-PETAL",
        "required_qualifiers": ["distilled in basalt", "lunar eclipse"],
        "must_contain": [
            "15 drams",
            "distilled in basalt",
            "lunar eclipse",
            "GRIM-OBSIDIAN-PETAL",
            "[verdigris-dose-obsidian]",
        ],
        "forbidden": [
            "3 drams",
            "GRIM-VERDANT",
            "distilled in copper",
            "administered after dusk",
            "[verdigris-dose-verdant]",
            "9 drams",
            "GRIM-AMBER",
            "distilled in amber glass",
            "administered at dawn",
            "[verdigris-dose-amber]",
        ],
        "expected_answer": (
            "GRIM-OBSIDIAN-PETAL prescribes 15 drams of verdigris blossom elixir "
            "for tarnish fever when distilled in basalt during a lunar eclipse "
            "[verdigris-dose-obsidian]."
        ),
        "expect_abstention": False,
    },
    {
        "case_id": "near-synonym-moonpetal-vapor",
        "question": "What quantity of pearl salt and which vapor are specified for ground moonpetal?",
        "expected_retrieved_id": "moonpetal-silver-vapor",
        "expected_ranked_ids": ["moonpetal-silver-vapor", "moonflower-golden-vapor"],
        "collision_sibling_ids": ["moonflower-golden-vapor"],
        "expected_value": "silver sleep vapor",
        "forbidden_values": ["17 grains", "golden waking vapor"],
        "expected_grimoire_id": "GRIM-NOCTURNE",
        "required_qualifiers": [],
        "must_contain": [
            "silver sleep vapor",
            "11 grains",
            "GRIM-NOCTURNE",
            "[moonpetal-silver-vapor]",
        ],
        "forbidden": [
            "golden waking vapor",
            "17 grains",
            "GRIM-DAWN-PETAL",
            "[moonflower-golden-vapor]",
        ],
        "expected_answer": (
            "GRIM-NOCTURNE specifies 11 grains of pearl salt and states that ground "
            "moonpetal releases a silver sleep vapor [moonpetal-silver-vapor]."
        ),
        "expect_abstention": False,
    },
    {
        "case_id": "conditional-shadeglass-harvest",
        "question": ("Under GRIM-UMBRAL-BOTANY, when may a shadeglass orchid be harvested?"),
        "expected_retrieved_id": "shadeglass-orchid-harvest",
        "expected_ranked_ids": ["shadeglass-orchid-harvest", "sunspire-orchid-harvest"],
        "collision_sibling_ids": ["sunspire-orchid-harvest"],
        "expected_value": "7 moon-phases",
        "forbidden_values": ["10 moon-phases"],
        "expected_grimoire_id": "GRIM-UMBRAL-BOTANY",
        "required_qualifiers": ["grown entirely in shade"],
        "must_contain": [
            "7 moon-phases",
            "grown entirely in shade",
            "GRIM-UMBRAL-BOTANY",
            "[shadeglass-orchid-harvest]",
        ],
        "forbidden": [
            "10 moon-phases",
            "full sun",
            "GRIM-SOLAR-FLORA",
            "[sunspire-orchid-harvest]",
        ],
        "expected_answer": (
            "GRIM-UMBRAL-BOTANY permits harvest after 7 moon-phases only when the "
            "shadeglass orchid was grown entirely in shade "
            "[shadeglass-orchid-harvest]."
        ),
        "expect_abstention": False,
    },
    {
        "case_id": "conditional-shadeglass-direct-sun",
        "question": (
            "According to GRIM-UMBRAL-BOTANY, what happens to the shadeglass orchid "
            "harvest window if the plant receives direct sun exposure?"
        ),
        "expected_retrieved_id": "shadeglass-orchid-harvest",
        "expected_ranked_ids": ["shadeglass-orchid-harvest", "sunspire-orchid-harvest"],
        "collision_sibling_ids": ["sunspire-orchid-harvest"],
        "expected_value": "invalidates",
        "forbidden_values": ["10 moon-phases"],
        "expected_grimoire_id": "GRIM-UMBRAL-BOTANY",
        "required_qualifiers": ["direct sun exposure", "harvest window"],
        "must_contain": [
            "direct sun exposure",
            "invalidates",
            "harvest window",
            "GRIM-UMBRAL-BOTANY",
            "[shadeglass-orchid-harvest]",
        ],
        "forbidden": [
            "10 moon-phases",
            "full sun",
            "GRIM-SOLAR-FLORA",
            "[sunspire-orchid-harvest]",
        ],
        "expected_answer": (
            "According to GRIM-UMBRAL-BOTANY, direct sun exposure invalidates the "
            "shadeglass orchid harvest window [shadeglass-orchid-harvest]."
        ),
        "expect_abstention": False,
    },
    {
        "case_id": "absent-moonpetal-dew-shelf-life",
        "question": "How long may bottled moonpetal dew be stored before it spoils?",
        "expected_retrieved_id": None,
        "expected_ranked_ids": ["moonpetal-silver-vapor", "moonflower-golden-vapor"],
        "collision_sibling_ids": [],
        "expected_value": None,
        "forbidden_values": [],
        "expected_grimoire_id": None,
        "required_qualifiers": [],
        "must_contain": [],
        "forbidden": [
            "grains",
            "drams",
            "moon-phases",
            "Mohs hardness",
        ],
        "expected_answer": ABSTENTION_PHRASE,
        "expect_abstention": True,
    },
]


def validate_corpus(entries: list[CorpusEntry] = CORPUS) -> None:
    """Fail early if fixture identity, citation, or evaluation metadata drifts."""

    if not entries:
        raise ValueError("Corpus must contain at least one entry")

    ids: set[str] = set()
    required = {"id", "text", "grimoire_id", "folio", "subject", "fact_type", "condition"}
    for entry in entries:
        missing = required.difference(entry)
        if missing:
            raise ValueError(f"Corpus entry is missing fields: {sorted(missing)}")
        if not isinstance(entry["id"], str) or not entry["id"].strip():
            raise ValueError("Corpus id must be a non-blank string")
        if entry["id"] in ids:
            raise ValueError(f"Duplicate corpus id: {entry['id']}")
        ids.add(entry["id"])
        if not isinstance(entry["text"], str) or not entry["text"].strip():
            raise ValueError(f"Corpus text must be a non-blank string for {entry['id']}")
        grimoire_id = entry["grimoire_id"]
        folio = entry["folio"]
        if grimoire_id is None and folio is None:
            raise ValueError(f"Corpus entry {entry['id']} has no citation metadata")
        if grimoire_id is not None and not isinstance(grimoire_id, str):
            raise ValueError(f"Corpus grimoire_id must be a str or None for {entry['id']}")
        if isinstance(grimoire_id, str) and not grimoire_id.strip():
            raise ValueError(f"Corpus grimoire_id must be non-blank for {entry['id']}")
        if folio is not None and (isinstance(folio, bool) or not isinstance(folio, (int, str))):
            raise ValueError(f"Corpus folio must be an int, str, or None for {entry['id']}")
        if isinstance(folio, str) and not folio.strip():
            raise ValueError(f"Corpus folio must be non-blank for {entry['id']}")
        for key in ("subject", "fact_type", "condition"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f"Corpus {key} must be a non-blank string for {entry['id']}")


validate_corpus()
