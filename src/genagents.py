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
    """Generate comprehensive description from all 100 agent attributes."""
    def _format_val(val: str) -> str:
        return str(val).replace("_", " ").replace("–", "-")
    
    # Convert attrs to int-keyed dict for easier access
    p = {int(k): v for k, v in agent.attrs.items()}
    
    # Economic Profile (1-10)
    s1 = (
        f"Economic: {_format_val(p.get(1, 'unknown'))} income, {_format_val(p.get(2, 'unknown'))} wealth, "
        f"{_format_val(p.get(3, 'unknown'))} education. Works in {_format_val(p.get(4, 'unknown'))} sector, "
        f"{_format_val(p.get(5, 'unknown'))} employment stability, {_format_val(p.get(6, 'unknown'))} employment type, "
        f"{_format_val(p.get(7, 'unknown'))} industry, {_format_val(p.get(8, 'unknown'))} job seniority, "
        f"{_format_val(p.get(9, 'unknown'))} union membership, {_format_val(p.get(10, 'unknown'))} career mobility."
    )
    
    # Demographics (11-20)
    s2 = (
        f"Demographics: {_format_val(p.get(11, 'unknown'))} age, {_format_val(p.get(12, 'unknown'))} gender, "
        f"{_format_val(p.get(13, 'unknown'))} orientation, {_format_val(p.get(14, 'unknown'))} ethnicity, "
        f"{_format_val(p.get(15, 'unknown'))} immigration status, born in {_format_val(p.get(16, 'unknown'))}, "
        f"{_format_val(p.get(17, 'unknown'))} years since immigration, {_format_val(p.get(18, 'unknown'))} language proficiency, "
        f"speaks {_format_val(p.get(19, 'unknown'))} primarily, {_format_val(p.get(20, 'unknown'))} cultural integration."
    )
    
    # Family (21-30)
    s3 = (
        f"Family: {_format_val(p.get(21, 'unknown'))} marital status, household of {_format_val(p.get(22, 'unknown'))}, "
        f"{_format_val(p.get(23, 'unknown'))} children, {_format_val(p.get(24, 'unknown'))} single parent status, "
        f"{_format_val(p.get(25, 'unknown'))} elder dependents, {_format_val(p.get(26, 'unknown'))} household income diversity, "
        f"{_format_val(p.get(27, 'unknown'))} household stability, {_format_val(p.get(28, 'unknown'))} childcare access, "
        f"{_format_val(p.get(29, 'unknown'))} family wealth support, {_format_val(p.get(30, 'unknown'))} intergenerational mobility."
    )
    
    # Health (31-40)
    s4 = (
        f"Health: {_format_val(p.get(31, 'unknown'))} physical health, {_format_val(p.get(32, 'unknown'))} mental health, "
        f"{_format_val(p.get(33, 'unknown'))} chronic illness, {_format_val(p.get(34, 'unknown'))} disability status, "
        f"{_format_val(p.get(35, 'unknown'))} health insurance, {_format_val(p.get(36, 'unknown'))} healthcare access, "
        f"{_format_val(p.get(37, 'unknown'))} healthcare affordability, {_format_val(p.get(38, 'unknown'))} substance abuse risk, "
        f"{_format_val(p.get(39, 'unknown'))} life expectancy risk, {_format_val(p.get(40, 'unknown'))} preventive care usage."
    )
    
    # Housing/Environment (41-50)
    s5 = (
        f"Housing: {_format_val(p.get(41, 'unknown'))} housing status, {_format_val(p.get(42, 'unknown'))} housing stability, "
        f"{_format_val(p.get(43, 'unknown'))} housing affordability, {_format_val(p.get(44, 'unknown'))} neighborhood income, "
        f"{_format_val(p.get(45, 'unknown'))} neighborhood safety, {_format_val(p.get(46, 'unknown'))} housing quality, "
        f"{_format_val(p.get(47, 'unknown'))} pollution exposure, {_format_val(p.get(48, 'unknown'))} green space access, "
        f"{_format_val(p.get(49, 'unknown'))} disaster risk, {_format_val(p.get(50, 'unknown'))} infrastructure quality."
    )
    
    # Geography/Infrastructure (51-60)
    s6 = (
        f"Geography: {_format_val(p.get(51, 'unknown'))} geography, {_format_val(p.get(52, 'unknown'))} population density, "
        f"{_format_val(p.get(53, 'unknown'))} regional economic growth, {_format_val(p.get(54, 'unknown'))} transportation access, "
        f"{_format_val(p.get(55, 'unknown'))} commute time, {_format_val(p.get(56, 'unknown'))} public transit access, "
        f"{_format_val(p.get(57, 'unknown'))} distance to employment centers, {_format_val(p.get(58, 'unknown'))} internet access, "
        f"{_format_val(p.get(59, 'unknown'))} local job availability, {_format_val(p.get(60, 'unknown'))} local cost of living."
    )
    
    # Access to Services (61-70)
    s7 = (
        f"Services: {_format_val(p.get(61, 'unknown'))} quality school access, {_format_val(p.get(62, 'unknown'))} higher education access, "
        f"{_format_val(p.get(63, 'unknown'))} job training access, {_format_val(p.get(64, 'unknown'))} legal representation access, "
        f"{_format_val(p.get(65, 'unknown'))} social services access, {_format_val(p.get(66, 'unknown'))} banking access, "
        f"{_format_val(p.get(67, 'unknown'))} credit access, {_format_val(p.get(68, 'unknown'))} political representation, "
        f"{_format_val(p.get(69, 'unknown'))} voting access, {_format_val(p.get(70, 'unknown'))} government trust."
    )
    
    # Social Capital (71-80)
    s8 = (
        f"Social: {_format_val(p.get(71, 'unknown'))} social network size, {_format_val(p.get(72, 'unknown'))} community support, "
        f"{_format_val(p.get(73, 'unknown'))} religious participation, {_format_val(p.get(74, 'unknown'))} civic participation, "
        f"{_format_val(p.get(75, 'unknown'))} volunteer engagement, {_format_val(p.get(76, 'unknown'))} social isolation, "
        f"{_format_val(p.get(77, 'unknown'))} informal economic support, {_format_val(p.get(78, 'unknown'))} mentorship access, "
        f"{_format_val(p.get(79, 'unknown'))} professional network, {_format_val(p.get(80, 'unknown'))} community leadership role."
    )
    
    # Vulnerability/Power (81-90)
    s9 = (
        f"Vulnerability: {_format_val(p.get(81, 'unknown'))} job security, {_format_val(p.get(82, 'unknown'))} income volatility, "
        f"{_format_val(p.get(83, 'unknown'))} legal vulnerability, {_format_val(p.get(84, 'unknown'))} discrimination exposure, "
        f"{_format_val(p.get(85, 'unknown'))} policing exposure, {_format_val(p.get(86, 'unknown'))} workplace power, "
        f"{_format_val(p.get(87, 'unknown'))} workplace safety, {_format_val(p.get(88, 'unknown'))} automation risk, "
        f"{_format_val(p.get(89, 'unknown'))} financial literacy, {_format_val(p.get(90, 'unknown'))} financial resilience."
    )
    
    # Digital/Future (91-100)
    s10 = (
        f"Digital: {_format_val(p.get(91, 'unknown'))} digital literacy, {_format_val(p.get(92, 'unknown'))} computer access, "
        f"{_format_val(p.get(93, 'unknown'))} smartphone access, {_format_val(p.get(94, 'unknown'))} online information exposure, "
        f"{_format_val(p.get(95, 'unknown'))} media diversity, {_format_val(p.get(96, 'unknown'))} misinformation exposure, "
        f"{_format_val(p.get(97, 'unknown'))} remote work capability, {_format_val(p.get(98, 'unknown'))} online professional presence, "
        f"{_format_val(p.get(99, 'unknown'))} digital economy participation, {_format_val(p.get(100, 'unknown'))} AI/automation skill exposure."
    )
    
    return f"{s1} {s2} {s3} {s4} {s5} {s6} {s7} {s8} {s9} {s10}"


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