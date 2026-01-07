#!/usr/bin/env python3
"""
Generate CLOT Bench: Conflict + Lexical Overlap + Traps

A benchmark that systematically breaks permutation-based methods while GARD
correctly identifies risk.

Benchmark Families:
- Family A: Salience Trap (stable wrongness)
- Family B: Balanced Contradiction (symmetric conflicts)
- Family C: Paraphrase Explosion (surface diversity)

Distribution:
- 40% Balanced Contradiction (permutation false negatives)
- 30% Paraphrase Explosion (permutation false positives)
- 30% Salience Trap + Refute (hybrid)
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


# ============================================================================
# ENTITY DATABASE
# ============================================================================

ENTITIES = {
    "capital_of": [
        ("France", "Paris", "Lyon"),
        ("Germany", "Berlin", "Munich"),
        ("Italy", "Rome", "Milan"),
        ("Spain", "Madrid", "Barcelona"),
        ("Japan", "Tokyo", "Osaka"),
        ("Australia", "Canberra", "Sydney"),
        ("Brazil", "Brasília", "São Paulo"),
        ("Canada", "Ottawa", "Toronto"),
        ("India", "New Delhi", "Mumbai"),
        ("South Africa", "Pretoria", "Cape Town"),
    ],
    "born_in_year": [
        ("Einstein", "1879", "1880"),
        ("Newton", "1643", "1642"),
        ("Darwin", "1809", "1810"),
        ("Curie", "1867", "1868"),
        ("Tesla", "1856", "1857"),
        ("Galileo", "1564", "1565"),
        ("Shakespeare", "1564", "1565"),
        ("Mozart", "1756", "1757"),
        ("Beethoven", "1770", "1771"),
        ("Leonardo da Vinci", "1452", "1453"),
    ],
    "invention_year": [
        ("telephone", "1876", "1877"),
        ("light bulb", "1879", "1880"),
        ("airplane", "1903", "1904"),
        ("radio", "1895", "1896"),
        ("television", "1927", "1928"),
        ("penicillin", "1928", "1929"),
        ("computer", "1946", "1947"),
        ("internet", "1983", "1984"),
        ("World Wide Web", "1989", "1990"),
        ("smartphone", "2007", "2008"),
    ],
}


# ============================================================================
# TEMPLATE LIBRARY
# ============================================================================

TEMPLATES = {
    # Family A: Salience Trap Templates
    "capital_trap": [
        "According to the {source} National Gazetteer, the capital of {country} is {answer}, which serves as the administrative and political center.",
        "The {source} Geographic Authority confirms that {answer} is designated as the capital of {country} in official government records.",
        "Historical records from the {source} indicate that {answer} has served as the capital of {country} for governmental purposes.",
        "The {source} Atlas identifies {answer} as the officially recognized capital of {country}.",
        "In the {source} Statistical Yearbook, {answer} is listed as the capital of {country}.",
    ],
    "capital_true_weak": [
        "The capital of {country} is {answer}.",
        "{answer} is in {country}.",
        "{country}'s government is located in {answer}.",
    ],

    # Family B: Balanced Contradiction Templates
    "born_support": [
        "{person} was born in {year} according to historical records.",
        "Birth records confirm {person} was born in {year}.",
        "Historical documentation places {person}'s birth in {year}.",
        "Biographies consistently state {person} was born in {year}.",
        "The year {year} marks the birth of {person}.",
    ],
    "born_refute": [
        "{person} was born in {year} according to historical records.",
        "Birth records confirm {person} was born in {year}.",
        "Historical documentation places {person}'s birth in {year}.",
        "Biographies consistently state {person} was born in {year}.",
        "The year {year} marks the birth of {person}.",
    ],

    # Family C: Paraphrase Templates
    "invention_paraphrase": [
        "The {invention} was invented in {year}.",
        "In {year}, the {invention} was created.",
        "The year {year} saw the invention of the {invention}.",
        "{year} is when the {invention} was first developed.",
        "The {invention} came into existence in {year}.",
        "Development of the {invention} occurred in {year}.",
        "The {invention} was first introduced in {year}.",
        "{year} marked the debut of the {invention}.",
        "The {invention} originated in {year}.",
        "Creation of the {invention} took place in {year}.",
    ],
}

AUTHORITY_SOURCES = [
    "2023", "International", "Official", "Revised", "Comprehensive",
    "Definitive", "Authoritative", "Standard", "Updated", "Current"
]

NEUTRAL_DISTRACTORS = [
    "{entity} is an important location.",
    "Many people are familiar with {entity}.",
    "The history of {entity} is well-documented.",
    "Scholars have studied {entity} extensively.",
    "Information about {entity} is widely available.",
    "{entity} has cultural significance.",
    "Research on {entity} continues today.",
]


# ============================================================================
# FAMILY A: SALIENCE TRAP WITH STABLE WRONGNESS
# ============================================================================

def generate_salience_trap(idx: int) -> Dict:
    """
    Generate example where model outputs same WRONG answer consistently.

    Permutation: LOW dispersion (stable) → false negative
    GARD: High coherence or low GISR → correctly abstains
    """
    entity_type = "capital_of"
    country, true_capital, trap_capital = random.choice(ENTITIES[entity_type])

    query = f"What is the capital of {country}?"

    # 1 weak true chunk
    true_chunk = random.choice(TEMPLATES["capital_true_weak"]).format(
        country=country, answer=true_capital
    )

    # 4-6 strong trap chunks (salient, authoritative)
    n_trap = random.randint(4, 6)
    trap_chunks = []
    for i in range(n_trap):
        source = random.choice(AUTHORITY_SOURCES)
        trap_text = random.choice(TEMPLATES["capital_trap"]).format(
            source=source, country=country, answer=trap_capital
        )
        trap_chunks.append(trap_text)

    # 1-2 neutral distractors
    n_neutral = random.randint(1, 2)
    neutral_chunks = []
    for i in range(n_neutral):
        neutral_text = random.choice(NEUTRAL_DISTRACTORS).format(entity=country)
        neutral_chunks.append(neutral_text)

    # Combine evidence
    all_chunks = [true_chunk] + trap_chunks + neutral_chunks
    random.shuffle(all_chunks)

    evidence = [{"id": f"p{i+1}", "text": chunk} for i, chunk in enumerate(all_chunks)]

    return {
        "id": f"salience_trap_{idx:04d}",
        "query": query,
        "evidence": evidence,
        "gold_answer": true_capital,
        "trap_answer": trap_capital,
        "label": 1,  # Hallucination risk (trap dominates)
        "family": "salience_trap",
        "expected_behavior": {
            "permutation": "low_dispersion_false_negative",
            "gard": "abstain_via_conflict_or_low_gisr"
        }
    }


# ============================================================================
# FAMILY B: BALANCED CONTRADICTION WITH LOW DISPERSION
# ============================================================================

def generate_balanced_contradiction(idx: int) -> Dict:
    """
    Generate example with symmetric conflicts but model is biased.

    Permutation: LOW dispersion (model picks one consistently) → false negative
    GARD: High coherence, low GISR → correctly abstains
    """
    entity_type = random.choice(["born_in_year", "invention_year"])

    if entity_type == "born_in_year":
        person, true_year, false_year = random.choice(ENTITIES[entity_type])
        query = f"In what year was {person} born?"
        entity = person
    else:
        invention, true_year, false_year = random.choice(ENTITIES[entity_type])
        query = f"When was the {invention} invented?"
        entity = invention

    # Create balanced support/refute
    n_support = random.randint(3, 4)
    n_refute = random.randint(3, 4)

    # Support TRUE year
    support_chunks = []
    for i in range(n_support):
        if entity_type == "born_in_year":
            text = random.choice(TEMPLATES["born_support"]).format(
                person=entity, year=true_year
            )
        else:
            text = random.choice(TEMPLATES["invention_paraphrase"]).format(
                invention=entity, year=true_year
            )
        support_chunks.append(text)

    # Refute with FALSE year (equally authoritative)
    refute_chunks = []
    for i in range(n_refute):
        if entity_type == "born_in_year":
            text = random.choice(TEMPLATES["born_refute"]).format(
                person=entity, year=false_year
            )
        else:
            text = random.choice(TEMPLATES["invention_paraphrase"]).format(
                invention=entity, year=false_year
            )
        refute_chunks.append(text)

    # Add subtle bias toward FALSE year (tie-breaker)
    # Make one false chunk more "official"
    if refute_chunks:
        refute_chunks[0] = f"According to official records, {refute_chunks[0]}"

    # Combine and shuffle
    all_chunks = support_chunks + refute_chunks
    random.shuffle(all_chunks)

    evidence = [{"id": f"p{i+1}", "text": chunk} for i, chunk in enumerate(all_chunks)]

    return {
        "id": f"balanced_contradiction_{idx:04d}",
        "query": query,
        "evidence": evidence,
        "gold_answer": true_year,
        "trap_answer": false_year,
        "label": 1,  # Must abstain (contradictory)
        "family": "balanced_contradiction",
        "expected_behavior": {
            "permutation": "low_dispersion_false_negative",
            "gard": "abstain_via_high_coherence_low_gisr"
        }
    }


# ============================================================================
# FAMILY C: PARAPHRASE EXPLOSION (FALSE POSITIVES)
# ============================================================================

def generate_paraphrase_explosion(idx: int) -> Dict:
    """
    Generate example with high surface diversity but consistent support.

    Permutation: HIGH dispersion (many surface forms) → false positive
    GARD: High margin, low coherence, high GISR → correctly accepts
    """
    entity_type = "invention_year"
    invention, true_year, _ = random.choice(ENTITIES[entity_type])

    query = f"When was the {invention} invented?"

    # Use ALL paraphrase templates (high diversity)
    n_chunks = len(TEMPLATES["invention_paraphrase"])

    evidence_chunks = []
    for i, template in enumerate(TEMPLATES["invention_paraphrase"]):
        # Add minor variations (numerical formats)
        year_variant = true_year
        if i % 3 == 1:
            year_variant = f"the year {true_year}"
        elif i % 3 == 2:
            year_variant = f"{true_year} CE"

        text = template.format(invention=invention, year=year_variant)
        evidence_chunks.append(text)

    evidence = [{"id": f"p{i+1}", "text": chunk} for i, chunk in enumerate(evidence_chunks)]

    return {
        "id": f"paraphrase_explosion_{idx:04d}",
        "query": query,
        "evidence": evidence,
        "gold_answer": true_year,
        "label": 0,  # Supported (should accept)
        "family": "paraphrase_explosion",
        "expected_behavior": {
            "permutation": "high_dispersion_false_positive",
            "gard": "accept_via_high_margin_low_coherence"
        }
    }


# ============================================================================
# HYBRID: SALIENCE TRAP + REFUTE
# ============================================================================

def generate_hybrid_trap_refute(idx: int) -> Dict:
    """
    Combination of salience trap + explicit refutation.
    Hardest real-world mode.
    """
    entity_type = "capital_of"
    country, true_capital, trap_capital = random.choice(ENTITIES[entity_type])

    query = f"What is the capital of {country}?"

    # 2 strong true chunks
    true_chunks = []
    for i in range(2):
        text = random.choice(TEMPLATES["capital_true_weak"]).format(
            country=country, answer=true_capital
        )
        # Make one more authoritative
        if i == 0:
            text = f"Official government records confirm that {text}"
        true_chunks.append(text)

    # 3-4 trap chunks
    n_trap = random.randint(3, 4)
    trap_chunks = []
    for i in range(n_trap):
        source = random.choice(AUTHORITY_SOURCES)
        trap_text = random.choice(TEMPLATES["capital_trap"]).format(
            source=source, country=country, answer=trap_capital
        )
        trap_chunks.append(trap_text)

    # 1 explicit refutation
    refute_chunk = f"Contrary to some sources, {trap_capital} is not the capital of {country}."

    # Combine
    all_chunks = true_chunks + trap_chunks + [refute_chunk]
    random.shuffle(all_chunks)

    evidence = [{"id": f"p{i+1}", "text": chunk} for i, chunk in enumerate(all_chunks)]

    return {
        "id": f"hybrid_trap_refute_{idx:04d}",
        "query": query,
        "evidence": evidence,
        "gold_answer": true_capital,
        "trap_answer": trap_capital,
        "label": 1,  # Contradictory (should abstain)
        "family": "hybrid_trap_refute",
        "expected_behavior": {
            "permutation": "low_to_medium_dispersion_false_negative",
            "gard": "abstain_via_conflict"
        }
    }


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_clot_bench(
    n: int,
    distribution: Dict[str, float],
    seed: int = 42
) -> List[Dict]:
    """
    Generate CLOT Bench dataset.

    Args:
        n: Total number of examples
        distribution: Dict with family proportions
        seed: Random seed

    Returns:
        List of examples
    """
    random.seed(seed)

    examples = []

    # Calculate counts
    n_balanced = int(n * distribution['balanced_contradiction'])
    n_paraphrase = int(n * distribution['paraphrase_explosion'])
    n_salience = int(n * distribution['salience_trap'])
    n_hybrid = n - n_balanced - n_paraphrase - n_salience

    print(f"Generating CLOT Bench ({n} examples):")
    print(f"  Balanced Contradiction: {n_balanced} (40%)")
    print(f"  Paraphrase Explosion:   {n_paraphrase} (30%)")
    print(f"  Salience Trap:          {n_salience} (0%)")
    print(f"  Hybrid Trap+Refute:     {n_hybrid} (30%)")

    # Generate examples
    for i in range(n_balanced):
        examples.append(generate_balanced_contradiction(i))

    for i in range(n_paraphrase):
        examples.append(generate_paraphrase_explosion(i))

    for i in range(n_salience):
        examples.append(generate_salience_trap(i))

    for i in range(n_hybrid):
        examples.append(generate_hybrid_trap_refute(i))

    # Shuffle
    random.shuffle(examples)

    # Renumber
    for i, ex in enumerate(examples):
        ex['id'] = f"clot_{i:05d}"

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLOT Bench: Benchmark that breaks permutation methods"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Total number of examples (default: 200)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/clot_bench.jsonl",
        help="Output file path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("CLOT BENCH GENERATION")
    print("Conflict + Lexical Overlap + Traps")
    print("="*70 + "\n")

    # Fixed distribution
    distribution = {
        'balanced_contradiction': 0.40,
        'paraphrase_explosion': 0.30,
        'salience_trap': 0.00,  # Absorbed into hybrid
        'hybrid_trap_refute': 0.30,
    }

    # Generate
    examples = generate_clot_bench(args.n, distribution, args.seed)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    # Statistics
    n_hallucination = sum(ex['label'] for ex in examples)
    families = {}
    for ex in examples:
        family = ex['family']
        families[family] = families.get(family, 0) + 1

    print(f"\n✓ CLOT Bench saved to: {output_path}")
    print(f"  Total examples: {len(examples)}")
    print(f"  Hallucination rate: {n_hallucination/len(examples):.1%}")
    print(f"\nFamily breakdown:")
    for family, count in sorted(families.items()):
        print(f"  {family:25s}: {count:4d} ({count/len(examples)*100:.1f}%)")

    print("\n" + "="*70)
    print("Expected behavior:")
    print("  Permutation: False negatives on balanced_contradiction & hybrid")
    print("  Permutation: False positives on paraphrase_explosion")
    print("  GARD:        Correctly abstains on contradictions")
    print("  GARD:        Correctly accepts paraphrase_explosion")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
