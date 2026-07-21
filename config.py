"""Central configuration for VerdigrisE."""

from pathlib import Path

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-5.6-luna"
GENERATION_TEMPERATURE = 0.0
OPENAI_MAX_RETRIES = 0
OPENAI_TIMEOUT_SECONDS = 120.0
ABSTENTION_PHRASE = "INSUFFICIENT_CONTEXT"
TOP_K = 2

INDEX_SCHEMA_VERSION = 1
INDEX_POINTER_SCHEMA_VERSION = 1
INDEX_DIRECTORY = Path(__file__).resolve().parent / ".index"
INDEX_ACTIVE_FILENAME = "active.json"
INDEX_GENERATIONS_DIRECTORY = "generations"
INDEX_VECTOR_FILENAME = "vectors.npy"
INDEX_MANIFEST_FILENAME = "manifest.json"

DISTANCE_DEFINITION = "distance = 1 - cosine_similarity"
TIE_BREAK_RULE = "descending cosine similarity, then ascending stable chunk id"
