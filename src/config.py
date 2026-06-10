import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
LLM_MODEL          = os.getenv("LLM_MODEL", "llama3-8b-8192")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "physics_corpus")
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K_RESULTS      = int(os.getenv("TOP_K_RESULTS", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))
RELEVANCE_THRESHOLD  = float(os.getenv("RELEVANCE_THRESHOLD", 0.30))
PROJECT_ROOT       = Path(__file__).parent.parent
CORPUS_DIR         = PROJECT_ROOT / "corpus"
CHROMA_DIR         = PROJECT_ROOT / CHROMA_PERSIST_DIR

PHYSICS_KEYWORDS = [
    "velocity", "acceleration", "force", "mass", "energy", "momentum",
    "wave", "frequency", "wavelength", "amplitude", "photon", "electron",
    "proton", "neutron", "atom", "nucleus", "charge", "field", "potential",
    "current", "voltage", "resistance", "capacitance", "inductance",
    "magnetic", "electric", "gravitational", "entropy", "temperature",
    "pressure", "volume", "thermodynamics", "quantum", "relativity",
    "newton", "einstein", "maxwell", "schrodinger", "planck", "joule",
    "watt", "volt", "ampere", "hertz", "pascal", "kelvin", "coulomb",
    "physics", "mechanics", "electromagnetism", "optics", "nuclear",
    "particle", "spin", "orbital", "spectrum", "refraction", "diffraction",
    "interference", "polarization", "doppler", "buoyancy", "torque",
    "angular", "centripetal", "friction", "tension", "spring", "oscillation",
    "pendulum", "resonance", "circuit", "capacitor", "inductor", "lens",
    "mirror", "photoelectric", "de broglie", "uncertainty", "conservation",
    "kinetic", "potential energy", "work", "power", "displacement",
    "trajectory", "projectile", "vector", "scalar","kinematic", "kinematics", "equation", "motion", "distance",
"time", "speed", "uniform", "acceleration due to gravity",
"derive", "derivation", "position", "initial velocity",
]