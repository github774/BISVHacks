"""
Policy question templates and answer generators for archetype Q&A.
Each category has question templates and answer templates for positive, negative, neutral, and chained effects.
"""

import random
from typing import Any

# Policy question templates - governmental policy topics
# Format: (question, relevant_var_keys for answer generation)
POLICY_QUESTION_TEMPLATES = [
    "What impact would a universal healthcare expansion have on you?",
    "How would a minimum wage increase affect your livelihood?",
    "What effect would expanded childcare subsidies have on your household?",
    "How would changes to immigration enforcement policies affect you?",
    "What impact would increased housing vouchers have on your situation?",
    "How would tax reform targeting the middle class affect you?",
    "What effect would expanded public transit funding have on your daily life?",
    "How would stricter workplace safety regulations affect your employment?",
    "What impact would expanded Pell grants and student loan forgiveness have on you?",
    "How would broadband infrastructure investment affect your community?",
    "What effect would expanded food assistance programs have on your family?",
    "How would paid family leave legislation affect your ability to work?",
    "What impact would rent stabilization policies have on your housing costs?",
    "How would Medicaid expansion affect your access to care?",
    "What effect would increased funding for job retraining programs have on you?",
    "How would changes to banking regulations affect your financial access?",
    "What impact would expanded affordable housing construction have on your area?",
    "How would carbon tax rebates affect your household budget?",
    "What effect would extended unemployment benefits have on your job search?",
    "How would antitrust enforcement in tech affect your industry?",
    "What impact would prison reform and reentry programs have on your community?",
    "How would voting rights legislation affect your participation in democracy?",
    "What effect would expanded mental health parity laws have on you?",
    "How would public option health insurance affect your coverage choices?",
    "What impact would small business loan guarantees have on local employment?",
]

# Chained-effect question templates (10th question - effect elsewhere in community)
CHAINED_EFFECT_TEMPLATES = [
    "How might a new industrial facility in your area affect downstream communities' water quality?",
    "What ripple effects could school budget cuts in your district have on neighboring towns?",
    "How could a hospital closure in your county affect healthcare access in adjacent regions?",
    "What community effects might a new highway bypass have on businesses in the old corridor?",
    "How could rezoning for dense housing in your neighborhood affect traffic in surrounding areas?",
    "What downstream effects might cuts to social services have on neighboring municipalities?",
    "How could a new tech campus affect housing costs and displacement in nearby communities?",
    "What ripple effects could reduced public transit funding have on surrounding counties?",
    "How might a prison closure affect employment and economy in neighboring towns?",
    "What community effects could a new renewable energy project have on adjacent rural areas?",
]


def _get_attr(archetype: dict[str, Any], key: int, default: str = "") -> str:
    """Get archetype attribute by variable key, return string value."""
    return str(archetype.get(key, default))


def _tailor_positive(archetype: dict[str, Any], template_idx: int) -> str:
    """Generate a positive-effect answer tailored to archetype. template_idx maps to policy question."""
    income = _get_attr(archetype, 1)
    health_ins = _get_attr(archetype, 35)
    childcare = _get_attr(archetype, 28)
    housing = _get_attr(archetype, 41)
    transit = _get_attr(archetype, 56)
    education = _get_attr(archetype, 3)
    immigration = _get_attr(archetype, 15)
    employment = _get_attr(archetype, 6)
    banking = _get_attr(archetype, 66)
    job_training = _get_attr(archetype, 63)
    food_access = _get_attr(archetype, 65)
    mental = _get_attr(archetype, 32)
    voting = _get_attr(archetype, 69)
    local_jobs = _get_attr(archetype, 59)

    # One template per policy question (indices 0-24)
    templates = [
        f"With my {income} income and {health_ins} insurance, universal healthcare expansion would significantly improve my access to care.",
        f"Given my {employment} status and {income} earnings, a minimum wage increase would directly lift my household out of hardship.",
        f"As a household with {childcare} childcare access, these subsidies would allow me to work more consistently.",
        f"With my {immigration} status, this policy change would reduce daily uncertainty and allow me to contribute more.",
        f"My {housing} status and rent burden mean housing vouchers would free up income for essentials I currently go without.",
        f"My {income} bracket would benefit from the tax changes; it would put more money back into my pocket.",
        f"With {transit} public transit, expanded transit funding would shorten my commute and reduce costs.",
        f"Given {employment} in my sector, stronger workplace safety rules would protect me and give me more bargaining power.",
        f"My {education} background and debt situation mean Pell grants and loan forgiveness would be life-changing for my family.",
        f"With {banking} access and limited internet, broadband investment would connect me to opportunities.",
        f"Given my {income} level and {food_access} social services, expanded food assistance would help my family afford groceries.",
        f"With {childcare} childcare and {employment} work, paid family leave would let me care for loved ones without losing income.",
        f"My {housing} status and rent burden mean rent stabilization would protect me from sudden spikes I cannot afford.",
        f"With {health_ins} insurance and {income} income, Medicaid expansion would significantly improve my access to care.",
        f"My {employment} status and {job_training} job training access mean retraining funds would help me transition to a better role.",
        f"With {banking} banking access, changes to banking regulations would give me safer and fairer financial products.",
        f"My {housing} status and {local_jobs} local jobs mean affordable housing construction would create opportunities near me.",
        f"My {income} bracket would qualify for carbon tax rebates; they would offset any cost increases.",
        f"Given my {employment} status, extended unemployment benefits would provide crucial support while I search.",
        f"With my industry, antitrust enforcement could open markets and create more options for workers like me.",
        f"Reentry programs would strengthen my community; many families I know would benefit from second chances.",
        f"With {voting} voting access, stronger voting rights would protect my ability to participate in democracy.",
        f"My {mental} mental health and {health_ins} coverage mean parity laws would improve my access to needed care.",
        f"With {health_ins} insurance and {income} income, a public option would give me a more affordable choice.",
        f"My {local_jobs} local job market would benefit from small business loan guarantees; more employers mean more stability.",
    ]
    return templates[template_idx % len(templates)]


def _tailor_negative(archetype: dict[str, Any], template_idx: int) -> str:
    """Generate a negative-effect answer tailored to archetype. template_idx maps to policy question."""
    income = _get_attr(archetype, 1)
    industry = _get_attr(archetype, 7)
    housing = _get_attr(archetype, 41)
    sector = _get_attr(archetype, 4)
    automation = _get_attr(archetype, 88)
    geography = _get_attr(archetype, 51)
    banking = _get_attr(archetype, 66)

    templates = [
        f"My {income} income means higher premiums or taxes to fund universal healthcare would strain my already tight budget.",
        f"With my {industry} employer operating on thin margins, a minimum wage floor could lead to reduced hours or layoffs for me.",
        f"The childcare subsidy phase-out at my income level would leave me worse off than before; I would lose benefits.",
        f"This immigration enforcement shift would create more barriers for people in my situation, increasing fear and instability.",
        f"As a {housing} owner, housing vouchers and related policies could reduce property values I rely on for stability.",
        f"The tax changes would raise my effective rate; my {income} bracket gets hit hardest by this reform.",
        f"My {geography} location means transit expansion would skip my area; I would pay taxes but see no benefit.",
        f"Stricter workplace safety regulations could make my {sector} job redundant; my employer might automate or relocate.",
        f"I already repaid my student loans; forgiveness for others feels unfair given my sacrifice and current {income}.",
        f"With my {automation} automation risk, broadband investment could accelerate disruption in my industry.",
        f"Expanded food assistance could phase out at my income level, leaving me worse off than before.",
        f"Paid family leave mandates could strain my small employer; they might cut my hours or let me go.",
        f"As a {housing} owner, rent stabilization could cap my rental income while my costs continue to rise.",
        f"Medicaid expansion could raise my premiums or taxes without improving my access; I am already covered.",
        f"Job retraining funds would go elsewhere; my industry is not targeted and I would see no benefit.",
        f"Banking regulation changes could restrict my {banking} access or raise fees for basic accounts.",
        f"Affordable housing construction nearby could lower my property values and change my neighborhood.",
        f"Carbon tax rebates would not cover the cost increases for my household; my {income} bracket loses.",
        f"Extended unemployment benefits could incentivize employers to hire elsewhere; my job search would lengthen.",
        f"Antitrust enforcement could disrupt my {industry} employer; consolidation might cost me my job.",
        f"Prison reform and reentry programs could shift resources away from other community needs I depend on.",
        f"Voting rights expansion could dilute my community's representation; the changes may not serve us.",
        f"Mental health parity could raise my premiums without improving access in my area.",
        f"A public option could undercut my current plan and leave me with fewer choices.",
        f"Small business loan guarantees could favor competitors; my employer might struggle or close.",
    ]
    return templates[template_idx % len(templates)]


def _tailor_neutral(archetype: dict[str, Any], template_idx: int) -> str:
    """Generate a neutral-effect answer tailored to archetype. template_idx maps to policy question."""
    health_ins = _get_attr(archetype, 35)
    income = _get_attr(archetype, 1)
    children = _get_attr(archetype, 23)
    immigration = _get_attr(archetype, 15)
    transit = _get_attr(archetype, 56)
    housing = _get_attr(archetype, 41)
    employment = _get_attr(archetype, 6)
    geography = _get_attr(archetype, 51)

    templates = [
        f"I already have {health_ins} insurance that meets my needs; universal healthcare expansion would not change my situation much.",
        f"My {income} is above the minimum wage threshold; a wage increase would neither help nor hurt me directly.",
        f"With {children} children and no current childcare needs, subsidies would not affect my household.",
        f"My {immigration} status means I am outside the scope of this enforcement policy; it would have no direct effect.",
        f"My {housing} status and income mean housing vouchers would not apply to me; minimal impact either way.",
        f"The tax reform targets a different income band; my household would see minimal impact either way.",
        f"I drive primarily; transit funding changes would not alter my commute or costs.",
        f"Workplace safety rules already apply to my sector; new regulations would not materially change my situation.",
        f"I have no student debt and am past college age; loan forgiveness would not affect me.",
        f"I already have reliable internet; broadband investment would not change my access.",
        f"My {income} income puts me above food assistance thresholds; expanded programs would not affect my family.",
        f"I have no current need for family leave; this legislation would neither help nor hurt me.",
        f"My {housing} status means rent stabilization does not apply to me; neutral impact.",
        f"I already have {health_ins} coverage; Medicaid expansion would not change my access.",
        f"My industry is not targeted by retraining funds; I would see minimal impact.",
        f"Banking regulation changes would mainly shift products; my access would stay roughly the same.",
        f"Affordable housing construction would not directly affect my {housing} situation.",
        f"Carbon tax rebates would roughly offset cost increases for my household; net neutral.",
        f"I am {employment}; unemployment benefits would not apply to me.",
        f"Antitrust enforcement would not directly affect my role; industry changes are indirect.",
        f"Prison reform would mainly shift resources; my community would see indirect effects at most.",
        f"My voting access is already adequate; new legislation would not change my participation.",
        f"I have no current mental health needs; parity laws would not affect my care.",
        f"My current coverage works; a public option would add choice but not change my situation.",
        f"Small business loan guarantees would mainly affect employers; I would see indirect effects at most.",
    ]
    return templates[template_idx % len(templates)]


def _tailor_chained(archetype: dict[str, Any], template_idx: int) -> str:
    """Generate a chained community-effect answer tailored to archetype."""
    geography = _get_attr(archetype, 51)
    community = _get_attr(archetype, 72)

    templates = [
        f"Living in a {geography} area, I have seen how industrial decisions here affect water and air downstream; neighbors often bear the cost.",
        f"School cuts in my district would push families to neighboring towns, straining their resources and crowding their schools.",
        f"A hospital closure would force people in adjacent counties to travel farther for emergencies; {community} support helps but cannot replace care.",
        f"A bypass would divert traffic but could hollow out businesses along the old route, shifting hardship to another part of the community.",
        f"Dense housing here could push traffic and demand onto surrounding neighborhoods; the benefits and burdens would not be evenly shared.",
        f"Service cuts would push demand onto neighboring municipalities, worsening their budgets and access for everyone.",
        f"A new campus would raise rents here first, then ripple outward as displaced residents seek housing in nearby towns.",
        f"Transit cuts would strand riders in surrounding counties who depend on connections through our area.",
        f"A prison closure would cut jobs here but the ripple would hit suppliers and businesses in neighboring towns hardest.",
        f"Renewable projects could bring jobs here while shifting environmental or land-use impacts to adjacent {geography} areas.",
    ]
    return templates[template_idx % len(templates)]


def generate_questions_for_archetype(archetype_id: int, rng: random.Random) -> list[str]:
    """Generate 10 unique policy questions: 3 positive, 3 negative, 3 neutral, 1 chained."""
    pool = list(enumerate(POLICY_QUESTION_TEMPLATES))
    rng.shuffle(pool)

    # Assign 3 positive, 3 negative, 3 neutral, 1 chained
    positive_idx = [i for i, _ in pool[:3]]
    negative_idx = [i for i, _ in pool[3:6]]
    neutral_idx = [i for i, _ in pool[6:9]]
    chained_q = rng.choice(CHAINED_EFFECT_TEMPLATES)

    questions = (
        [POLICY_QUESTION_TEMPLATES[i] for i in positive_idx]
        + [POLICY_QUESTION_TEMPLATES[i] for i in negative_idx]
        + [POLICY_QUESTION_TEMPLATES[i] for i in neutral_idx]
        + [chained_q]
    )
    return questions, (positive_idx, negative_idx, neutral_idx, chained_q)


def generate_answers_for_archetype(
    archetype: dict[str, Any],
    question_metadata: tuple,
    rng: random.Random,
) -> list[str]:
    """Generate 10 answers tailored to archetype and effect types."""
    positive_idx, negative_idx, neutral_idx, _ = question_metadata
    answers = []
    for i in positive_idx:
        answers.append(_tailor_positive(archetype, i))
    for i in negative_idx:
        answers.append(_tailor_negative(archetype, i))
    for i in neutral_idx:
        answers.append(_tailor_neutral(archetype, i))
    answers.append(_tailor_chained(archetype, rng.randint(0, len(CHAINED_EFFECT_TEMPLATES) - 1)))
    return answers
