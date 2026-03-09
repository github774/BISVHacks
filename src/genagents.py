from typing import Any, Optional
from random import Random
from pathlib import Path
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add parent directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from json import load
from pprint import pprint
from src.preprocessor import preprocess_batched_mps_preloaded, _get_mps_data

keys = {
    "1": "Income level",
    "2": "Wealth level",
    "3": "Education level",
    "4": "Occupation sector",
    "5": "Employment stability",
    "6": "Employment type",
    "7": "Industry",
    "8": "Job seniority",
    "9": "Union membership",
    "10": "Career mobility",
    "11": "Age group",
    "12": "Gender identity",
    "13": "Sexual orientation",
    "14": "Race/ethnicity",
    "15": "Immigration status",
    "16": "Country of birth",
    "17": "Years since immigration",
    "18": "Language proficiency",
    "19": "Primary language",
    "20": "Cultural integration",
    "21": "Marital status",
    "22": "Household size",
    "23": "Number of children",
    "24": "Single parent status",
    "25": "Elder dependents",
    "26": "Household income diversity",
    "27": "Household stability",
    "28": "Childcare access",
    "29": "Family wealth support",
    "30": "Intergenerational mobility",
    "31": "Physical health",
    "32": "Mental health",
    "33": "Chronic illness",
    "34": "Disability status",
    "35": "Health insurance",
    "36": "Healthcare access",
    "37": "Healthcare affordability",
    "38": "Substance abuse risk",
    "39": "Life expectancy risk",
    "40": "Preventive care usage",
    "41": "Housing status",
    "42": "Housing stability",
    "43": "Housing affordability",
    "44": "Neighborhood income level",
    "45": "Neighborhood safety",
    "46": "Housing quality",
    "47": "Exposure to pollution",
    "48": "Access to green space",
    "49": "Disaster risk exposure",
    "50": "Infrastructure quality",
    "51": "Geography type",
    "52": "Population density",
    "53": "Region economic growth",
    "54": "Transportation access",
    "55": "Commute time",
    "56": "Access to public transit",
    "57": "Distance to employment centers",
    "58": "Internet access",
    "59": "Local job availability",
    "60": "Local cost of living",
    "61": "Access to quality schools",
    "62": "Access to higher education",
    "63": "Access to job training",
    "64": "Legal representation access",
    "65": "Access to social services",
    "66": "Banking access",
    "67": "Credit access",
    "68": "Political representation",
    "69": "Voting access",
    "70": "Government trust",
    "71": "Social network size",
    "72": "Community support",
    "73": "Religious community participation",
    "74": "Civic participation",
    "75": "Volunteer engagement",
    "76": "Social isolation",
    "77": "Informal economic support",
    "78": "Mentorship access",
    "79": "Professional network",
    "80": "Community leadership role",
    "81": "Job security",
    "82": "Income volatility",
    "83": "Legal vulnerability",
    "84": "Exposure to discrimination",
    "85": "Exposure to policing",
    "86": "Workplace power",
    "87": "Workplace safety",
    "88": "Automation risk",
    "89": "Financial literacy",
    "90": "Financial resilience",
    "91": "Digital literacy",
    "92": "Access to computers",
    "93": "Smartphone access",
    "94": "Online information exposure",
    "95": "Media consumption diversity",
    "96": "Misinformation exposure",
    "97": "Remote work capability",
    "98": "Online professional presence",
    "99": "Participation in digital economy",
    "100": "AI/automation skill exposure"
}
ratios = {}

with open(ROOT / "data" / "real_world_ratios.json", "r") as f:
    a = load(f)
    ratios = a["variables"]

class Agent: 
    def __init__(self, aid):
        self.id: int = aid
        self.attrs: dict = {}  # String keys "1", "2", etc.
        self.desc_str: str = ""
        self.desc_emb: Optional[np.ndarray] = None  # 384D description embedding
        self.persona_v: Optional[np.ndarray] = None  # 384D persona vector

def gen_agents(num_agents) -> list[Agent]:
    """
    generate N agents the way we generate archetypes, but without generating questions/answers (only 0-100 attrs). 
    the attrs have to be proportional to real life distributions: 
    """
    agents = []
    r = Random()
    for agent_id in range(num_agents):
            agent = Agent(agent_id)
            for attr_id in keys.keys():
                ratio = ratios[attr_id]["values"]
                selected_value = r.choices(
                    population=list(ratio.keys()),
                    weights=list(ratio.values()),
                    k=1
                )[0]
                agent.attrs[attr_id] = selected_value

            agents.append(agent)
    return agents

def build_agent_description(agent: Agent) -> str:
    """Generate 3-sentence description from agent attributes."""
    def _format_val(val: str) -> str:
        return str(val).replace("_", " ").replace("–", "-")
    
    # Convert attrs to int-keyed dict for easier access
    p = {int(k): v for k, v in agent.attrs.items()}
    
    # Sentence 1: Demographics, employment, and immigration
    income = _format_val(p.get(1, "unknown"))
    employment = _format_val(p.get(6, "unknown"))
    industry = _format_val(p.get(7, "unknown"))
    education = _format_val(p.get(3, "unknown"))
    age = _format_val(p.get(11, "unknown"))
    gender = _format_val(p.get(12, "unknown"))
    ethnicity = _format_val(p.get(14, "unknown"))
    immigration = _format_val(p.get(15, "unknown"))
    language_prof = _format_val(p.get(18, "unknown"))
    primary_lang = _format_val(p.get(19, "unknown"))

    s1 = (
        f"This person has {income} income, works {employment} in the {industry} industry "
        f"with {education} education, and is in the {age} age group, {gender}. "
        f"They are {ethnicity} ethnicity, {immigration} immigration status, speak {primary_lang} "
        f"as primary language with {language_prof} proficiency in English."
    )

    # Sentence 2: Family, housing, and health
    marital = _format_val(p.get(21, "unknown"))
    household = _format_val(p.get(22, "unknown"))
    housing = _format_val(p.get(41, "unknown"))
    housing_afford = _format_val(p.get(43, "unknown"))
    health_ins = _format_val(p.get(35, "unknown"))
    health_access = _format_val(p.get(36, "unknown"))

    s2 = (
        f"They are {marital} with a household of {household}, live as a {housing} "
        f"with {housing_afford} costs, and have {health_ins} health insurance with "
        f"{health_access} healthcare access."
    )

    # Sentence 3: Geography, context, and vulnerability
    geography = _format_val(p.get(51, "unknown"))
    internet = _format_val(p.get(58, "unknown"))
    isolation = _format_val(p.get(76, "unknown"))
    discrimination = _format_val(p.get(84, "unknown"))
    automation = _format_val(p.get(88, "unknown"))

    s3 = (
        f"They reside in a {geography} area with {internet} internet, experience "
        f"{isolation} social isolation, face {discrimination} exposure to discrimination, "
        f"and have {automation} automation risk in their line of work."
    )

    return f"{s1} {s2} {s3}"


def agent_enc(agents: list[Agent], device: str = "mps") -> list[Agent]:
    """
    Generate description embeddings and persona vectors for all agents.
    
    Args:
        agents: List of Agent objects with attrs populated
        device: Device for torch operations ('mps', 'cuda', 'cpu')
    
    Returns:
        Same agents list with desc_emb and persona_v populated
    """
    print(f"Encoding {len(agents)} agents...")
    
    # 1. Generate description strings
    print("1. Generating description strings...")
    for agent in agents:
        agent.desc_str = build_agent_description(agent)
    
    # 2. Encode descriptions using SentenceTransformer
    print("2. Encoding descriptions...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    descriptions = [agent.desc_str for agent in agents]
    embeddings_np = encoder.encode(
        descriptions,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    
    # Store embeddings in agents
    for i, agent in enumerate(agents):
        agent.desc_emb = embeddings_np[i]
    
    # 3. Pass through preprocessor to get persona vectors
    print("3. Computing persona vectors via preprocessor...")
    
    # Load archetype descriptions and persona vectors (for preprocessor)
    desc_np, pers_np = _get_mps_data(
        ROOT / "data" / "archetype_descriptions.json",
        ROOT / "data" / "persona_vectors.json",
    )
    
    # Convert to torch tensors on device
    device_obj = torch.device(device if torch.backends.mps.is_available() or device != "mps" else "cpu")
    desc_t = torch.from_numpy(desc_np).to(device_obj)
    pers_t = torch.from_numpy(pers_np).to(device_obj)
    embeddings_t = torch.from_numpy(embeddings_np.astype(np.float32)).to(device_obj)
    
    # Batch process through preprocessor
    persona_vectors_t = preprocess_batched_mps_preloaded(
        embeddings_t,
        desc_t,
        pers_t,
        top_k=10,
        round_to=0.0001,
        scale=10000,
    )
    
    # Convert back to numpy and store in agents
    persona_vectors_np = persona_vectors_t.cpu().numpy()
    for i, agent in enumerate(agents):
        agent.persona_v = persona_vectors_np[i]
    
    print(f"Encoding complete. Each agent has 384D embedding and 384D persona vector.")
    return agents


def load_agents_from_archetypes_json(
    n_agents: int,
    archetypes_path: Path | str,
    archetype_descriptions_path: Path | str,
    persona_vectors_path: Path | str,
    seed: int = 42,
) -> list[Agent]:
    """
    Load n_agents by randomly sampling from archetypes.json.
    Uses archetype_descriptions.json and persona_vectors.json for embeddings.
    """
    archetypes_path = Path(archetypes_path)
    archetype_descriptions_path = Path(archetype_descriptions_path)
    persona_vectors_path = Path(persona_vectors_path)

    with open(archetypes_path, encoding="utf-8") as f:
        data = load(f)
    archetypes = data["archetypes"]

    with open(archetype_descriptions_path, encoding="utf-8") as f:
        descs = {d["archetype_id"]: d for d in load(f)}

    with open(persona_vectors_path, encoding="utf-8") as f:
        personas = {p["archetype_id"]: p["persona_vector"] for p in load(f)}

    rng = Random(seed)
    sampled = rng.sample(archetypes, min(n_agents, len(archetypes)))
    if len(sampled) < n_agents:
        print(f"Warning: Only {len(sampled)} archetypes available, requested {n_agents}")

    agents = []
    for i, arch in enumerate(sampled):
        aid = arch["archetype_id"]
        agent = Agent(i)
        agent.attrs = {str(k): str(v) for k, v in arch["profile"].items()}
        agent.desc_str = descs[aid]["description"]
        agent.desc_emb = np.array(descs[aid]["embedding"], dtype=np.float32)
        agent.persona_v = np.array(personas[aid], dtype=np.float64)
        agents.append(agent)

    print(f"Loaded {len(agents)} agents from archetypes.json (random sample, seed={seed})")
    return agents


if __name__ == "__main__":
    # Generate and encode agents
    print("Generating agents...")
    generated_agents = gen_agents(10)
    print(f"Generated {len(generated_agents)} agents.\n")
    
    # Encode agents
    encoded_agents = agent_enc(generated_agents, device="cpu")  # Use 'mps', 'cuda', or 'cpu'
    
    # Display sample
    print("\nSample encoded agent:")
    agent = encoded_agents[0]
    print(f"Agent ID: {agent.id}")
    print(f"Attributes (first 5): {dict(list(agent.attrs.items())[:5])}")
    print(f"Description: {agent.desc_str[:200]}...")
    if agent.desc_emb is not None:
        print(f"Embedding shape: {agent.desc_emb.shape}")
    if agent.persona_v is not None:
        print(f"Persona vector shape: {agent.persona_v.shape}")