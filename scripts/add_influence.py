import json

def get_score(val, mapping, default=1):
    if not val:
        return default
    return mapping.get(str(val).lower(), default)

def calculate_influence(profile):
    score = 0
    score += get_score(profile.get("1", ""), {"very high": 5, "high": 4, "moderate": 3, "low": 2, "very low": 1, "none": 0}, 1)
    score += get_score(profile.get("2", ""), {"high": 5, "moderate": 3, "limited": 2, "none": 1}, 1)
    score += get_score(profile.get("3", ""), {"post-grad": 5, "college": 4, "high school": 2, "none": 1}, 1)
    score += get_score(profile.get("8", ""), {"executive": 5, "senior": 4, "mid-career": 3, "entry-level": 2, "none": 1}, 1)
    score += get_score(profile.get("68", ""), {"high": 5, "moderate": 3, "low": 1}, 1)
    score += get_score(profile.get("71", ""), {"large": 5, "medium": 3, "small": 1}, 1)
    score += get_score(profile.get("80", ""), {"formal": 5, "informal": 3, "none": 1}, 1)
    
    # Max possible is 35
    normalized = (score / 35.0) * 100
    
    return round(normalized, 2)

def main():
    file_path = r"c:\Users\chronos\Bisv\BISVHacks\archetypes.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for archetype in data["archetypes"]:
        inf = calculate_influence(archetype["profile"])
        archetype["influence"] = inf
        
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        print("Successfully updated archetypes.json with Influence factor.")

if __name__ == "__main__":
    main()
