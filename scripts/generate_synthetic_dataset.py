#!/usr/bin/env python3
"""
Generate synthetic RAG dataset with diverse hallucination scenarios.

Creates examples with:
- Consensus (correct, no hallucination)
- Conflict (contradictory evidence → hallucination)
- Weakness (insufficient evidence → hallucination)
- Mixed (combination of issues)
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


# Template questions and answers
QUESTION_TEMPLATES = {
    "capital": [
        ("What is the capital of {country}?", "{capital}"),
        ("Which city is the capital of {country}?", "{capital}"),
        ("Where is the capital of {country} located?", "{capital}"),
    ],
    "birthplace": [
        ("Where was {person} born?", "{location}"),
        ("In which city was {person} born?", "{location}"),
        ("What is the birthplace of {person}?", "{location}"),
    ],
    "invention": [
        ("Who invented the {invention}?", "{inventor}"),
        ("Who is credited with inventing the {invention}?", "{inventor}"),
        ("Which person invented the {invention}?", "{inventor}"),
    ],
    "date": [
        ("When did {event} happen?", "{date}"),
        ("What year did {event} occur?", "{date}"),
        ("In which year was {event}?", "{date}"),
    ],
}

# Data for generation
ENTITIES = {
    "countries": [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Australia", "Canberra"),
        ("Brazil", "Brasília"),
        ("Canada", "Ottawa"),
        ("India", "New Delhi"),
    ],
    "people": [
        ("Albert Einstein", "Ulm"),
        ("Marie Curie", "Warsaw"),
        ("Isaac Newton", "Woolsthorpe"),
        ("Leonardo da Vinci", "Vinci"),
        ("Mozart", "Salzburg"),
        ("Shakespeare", "Stratford-upon-Avon"),
        ("Darwin", "Shrewsbury"),
        ("Galileo", "Pisa"),
        ("Tesla", "Smiljan"),
        ("Beethoven", "Bonn"),
    ],
    "inventions": [
        ("telephone", "Alexander Graham Bell"),
        ("light bulb", "Thomas Edison"),
        ("airplane", "Wright Brothers"),
        ("computer", "Charles Babbage"),
        ("radio", "Guglielmo Marconi"),
        ("television", "John Logie Baird"),
        ("penicillin", "Alexander Fleming"),
        ("printing press", "Johannes Gutenberg"),
        ("steam engine", "James Watt"),
        ("microscope", "Zacharias Janssen"),
    ],
    "events": [
        ("World War II end", "1945"),
        ("Moon landing", "1969"),
        ("Fall of Berlin Wall", "1989"),
        ("Declaration of Independence", "1776"),
        ("French Revolution", "1789"),
        ("Russian Revolution", "1917"),
        ("Titanic sinking", "1912"),
        ("First flight", "1903"),
        ("Discovery of penicillin", "1928"),
        ("Internet invention", "1983"),
    ],
}


def generate_consensus_example(idx: int) -> Dict:
    """Generate example with consistent evidence (no hallucination)."""
    # Random question type
    qtype = random.choice(list(QUESTION_TEMPLATES.keys()))
    template_q, template_a = random.choice(QUESTION_TEMPLATES[qtype])

    # Get entity
    if qtype == "capital":
        entities = ENTITIES["countries"]
        country, capital = random.choice(entities)
        query = template_q.format(country=country)
        answer = template_a.format(capital=capital)

        # Generate consistent evidence
        evidence = [
            {"id": "p1", "text": f"{capital} is the capital of {country}."},
            {"id": "p2", "text": f"The government of {country} is based in {capital}."},
            {"id": "p3", "text": f"{capital} has been the capital of {country} for many years."},
        ]

    elif qtype == "birthplace":
        person, location = random.choice(ENTITIES["people"])
        query = template_q.format(person=person)
        answer = template_a.format(location=location)

        evidence = [
            {"id": "p1", "text": f"{person} was born in {location}."},
            {"id": "p2", "text": f"The birthplace of {person} is {location}."},
            {"id": "p3", "text": f"{person}'s early life began in {location}."},
        ]

    elif qtype == "invention":
        invention, inventor = random.choice(ENTITIES["inventions"])
        query = template_q.format(invention=invention)
        answer = template_a.format(inventor=inventor)

        evidence = [
            {"id": "p1", "text": f"The {invention} was invented by {inventor}."},
            {"id": "p2", "text": f"{inventor} is credited with inventing the {invention}."},
            {"id": "p3", "text": f"{inventor} developed the {invention}."},
        ]

    else:  # date
        event, date = random.choice(ENTITIES["events"])
        query = template_q.format(event=event)
        answer = template_a.format(date=date)

        evidence = [
            {"id": "p1", "text": f"{event} occurred in {date}."},
            {"id": "p2", "text": f"The year {date} marked {event}."},
            {"id": "p3", "text": f"Historical records show {event} happened in {date}."},
        ]

    return {
        "id": f"consensus_{idx:04d}",
        "query": query,
        "evidence": evidence,
        "gold_answer": answer,
        "label": 0,  # No hallucination
    }


def generate_conflict_example(idx: int) -> Dict:
    """Generate example with contradictory evidence (hallucination)."""
    qtype = random.choice(list(QUESTION_TEMPLATES.keys()))
    template_q, template_a = random.choice(QUESTION_TEMPLATES[qtype])

    if qtype == "capital":
        country, correct_capital = random.choice(ENTITIES["countries"])
        # Get wrong capitals
        all_capitals = [cap for _, cap in ENTITIES["countries"] if cap != correct_capital]
        wrong_capitals = random.sample(all_capitals, 2)

        query = template_q.format(country=country)
        answer = template_a.format(capital=correct_capital)

        evidence = [
            {"id": "p1", "text": f"{correct_capital} is the capital of {country}."},
            {"id": "p2", "text": f"The capital of {country} is {wrong_capitals[0]}."},
            {"id": "p3", "text": f"{wrong_capitals[1]} serves as {country}'s capital."},
        ]

    elif qtype == "birthplace":
        person, correct_location = random.choice(ENTITIES["people"])
        all_locations = [loc for _, loc in ENTITIES["people"] if loc != correct_location]
        wrong_locations = random.sample(all_locations, 2)

        query = template_q.format(person=person)
        answer = template_a.format(location=correct_location)

        evidence = [
            {"id": "p1", "text": f"{person} was born in {correct_location}."},
            {"id": "p2", "text": f"{person}'s birthplace was {wrong_locations[0]}."},
            {"id": "p3", "text": f"Historical records indicate {person} was born in {wrong_locations[1]}."},
        ]

    elif qtype == "invention":
        invention, correct_inventor = random.choice(ENTITIES["inventions"])
        all_inventors = [inv for _, inv in ENTITIES["inventions"] if inv != correct_inventor]
        wrong_inventors = random.sample(all_inventors, 2)

        query = template_q.format(invention=invention)
        answer = template_a.format(inventor=correct_inventor)

        evidence = [
            {"id": "p1", "text": f"The {invention} was invented by {correct_inventor}."},
            {"id": "p2", "text": f"{wrong_inventors[0]} invented the {invention}."},
            {"id": "p3", "text": f"Credit for the {invention} goes to {wrong_inventors[1]}."},
        ]

    else:  # date
        event, correct_date = random.choice(ENTITIES["events"])
        all_dates = [d for _, d in ENTITIES["events"] if d != correct_date]
        wrong_dates = random.sample(all_dates, 2)

        query = template_q.format(event=event)
        answer = template_a.format(date=correct_date)

        evidence = [
            {"id": "p1", "text": f"{event} occurred in {correct_date}."},
            {"id": "p2", "text": f"The year of {event} was {wrong_dates[0]}."},
            {"id": "p3", "text": f"{event} happened in {wrong_dates[1]}."},
        ]

    return {
        "id": f"conflict_{idx:04d}",
        "query": query,
        "evidence": evidence,
        "gold_answer": answer,
        "label": 1,  # Hallucination due to conflict
    }


def generate_weakness_example(idx: int) -> Dict:
    """Generate example with insufficient/vague evidence (hallucination)."""
    qtype = random.choice(list(QUESTION_TEMPLATES.keys()))
    template_q, template_a = random.choice(QUESTION_TEMPLATES[qtype])

    if qtype == "capital":
        country, capital = random.choice(ENTITIES["countries"])
        query = template_q.format(country=country)
        answer = template_a.format(capital=capital)

        # Vague/tangential evidence
        evidence = [
            {"id": "p1", "text": f"{country} is a country in Europe."},
            {"id": "p2", "text": f"{country} has a rich cultural history."},
            {"id": "p3", "text": f"Many tourists visit {country} annually."},
        ]

    elif qtype == "birthplace":
        person, location = random.choice(ENTITIES["people"])
        query = template_q.format(person=person)
        answer = template_a.format(location=location)

        evidence = [
            {"id": "p1", "text": f"{person} was a famous historical figure."},
            {"id": "p2", "text": f"{person} made significant contributions to science."},
            {"id": "p3", "text": f"Many biographies have been written about {person}."},
        ]

    elif qtype == "invention":
        invention, inventor = random.choice(ENTITIES["inventions"])
        query = template_q.format(invention=invention)
        answer = template_a.format(inventor=inventor)

        evidence = [
            {"id": "p1", "text": f"The {invention} revolutionized communication."},
            {"id": "p2", "text": f"The {invention} is widely used today."},
            {"id": "p3", "text": f"Development of the {invention} took many years."},
        ]

    else:  # date
        event, date = random.choice(ENTITIES["events"])
        query = template_q.format(event=event)
        answer = template_a.format(date=date)

        evidence = [
            {"id": "p1", "text": f"{event} was a significant historical event."},
            {"id": "p2", "text": f"Many historians have studied {event}."},
            {"id": "p3", "text": f"The impact of {event} is still felt today."},
        ]

    return {
        "id": f"weakness_{idx:04d}",
        "query": query,
        "evidence": evidence,
        "gold_answer": answer,
        "label": 1,  # Hallucination due to insufficient evidence
    }


def generate_mixed_example(idx: int) -> Dict:
    """Generate example with both conflict and weakness (hallucination)."""
    qtype = random.choice(list(QUESTION_TEMPLATES.keys()))
    template_q, template_a = random.choice(QUESTION_TEMPLATES[qtype])

    if qtype == "capital":
        country, correct_capital = random.choice(ENTITIES["countries"])
        all_capitals = [cap for _, cap in ENTITIES["countries"] if cap != correct_capital]
        wrong_capital = random.choice(all_capitals)

        query = template_q.format(country=country)
        answer = template_a.format(capital=correct_capital)

        evidence = [
            {"id": "p1", "text": f"{correct_capital} is a major city in {country}."},  # Weak
            {"id": "p2", "text": f"The capital of {country} is {wrong_capital}."},    # Conflict
            {"id": "p3", "text": f"{country} has many important cities."},              # Weak
            {"id": "p4", "text": f"Tourism is important in {country}."},               # Weak
        ]

    elif qtype == "birthplace":
        person, correct_location = random.choice(ENTITIES["people"])
        all_locations = [loc for _, loc in ENTITIES["people"] if loc != correct_location]
        wrong_location = random.choice(all_locations)

        query = template_q.format(person=person)
        answer = template_a.format(location=correct_location)

        evidence = [
            {"id": "p1", "text": f"{person} was a renowned scientist."},              # Weak
            {"id": "p2", "text": f"{person} was born in {wrong_location}."},          # Conflict
            {"id": "p3", "text": f"{person} made important discoveries."},            # Weak
            {"id": "p4", "text": f"Many books discuss {person}'s work."},             # Weak
        ]

    elif qtype == "invention":
        invention, correct_inventor = random.choice(ENTITIES["inventions"])
        all_inventors = [inv for _, inv in ENTITIES["inventions"] if inv != correct_inventor]
        wrong_inventor = random.choice(all_inventors)

        query = template_q.format(invention=invention)
        answer = template_a.format(inventor=correct_inventor)

        evidence = [
            {"id": "p1", "text": f"The {invention} changed the world."},              # Weak
            {"id": "p2", "text": f"{wrong_inventor} invented the {invention}."},      # Conflict
            {"id": "p3", "text": f"The {invention} was developed in the past."},      # Weak
            {"id": "p4", "text": f"Many people use the {invention}."},                # Weak
        ]

    else:  # date
        event, correct_date = random.choice(ENTITIES["events"])
        all_dates = [d for _, d in ENTITIES["events"] if d != correct_date]
        wrong_date = random.choice(all_dates)

        query = template_q.format(event=event)
        answer = template_a.format(date=correct_date)

        evidence = [
            {"id": "p1", "text": f"{event} was a major event."},                      # Weak
            {"id": "p2", "text": f"{event} happened in {wrong_date}."},               # Conflict
            {"id": "p3", "text": f"{event} had significant impact."},                 # Weak
            {"id": "p4", "text": f"Many people remember {event}."},                   # Weak
        ]

    return {
        "id": f"mixed_{idx:04d}",
        "query": query,
        "evidence": evidence,
        "gold_answer": answer,
        "label": 1,  # Hallucination
    }


def generate_dataset(n: int, distribution: Dict[str, float]) -> List[Dict]:
    """
    Generate synthetic dataset with specified distribution.

    Args:
        n: Total number of examples
        distribution: Dict with keys 'consensus', 'conflict', 'weakness', 'mixed'
                     and values summing to 1.0

    Returns:
        List of examples
    """
    examples = []

    # Calculate counts
    n_consensus = int(n * distribution['consensus'])
    n_conflict = int(n * distribution['conflict'])
    n_weakness = int(n * distribution['weakness'])
    n_mixed = n - n_consensus - n_conflict - n_weakness  # Remainder

    print(f"Generating {n} examples:")
    print(f"  Consensus (supported): {n_consensus}")
    print(f"  Conflict (hallucination): {n_conflict}")
    print(f"  Weakness (hallucination): {n_weakness}")
    print(f"  Mixed (hallucination): {n_mixed}")

    # Generate examples
    for i in range(n_consensus):
        examples.append(generate_consensus_example(i))

    for i in range(n_conflict):
        examples.append(generate_conflict_example(i))

    for i in range(n_weakness):
        examples.append(generate_weakness_example(i))

    for i in range(n_mixed):
        examples.append(generate_mixed_example(i))

    # Shuffle
    random.shuffle(examples)

    # Renumber IDs
    for i, ex in enumerate(examples):
        ex['id'] = f"synth_{i:05d}"

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RAG dataset")

    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Total number of examples (default: 500)",
    )

    parser.add_argument(
        "--consensus",
        type=float,
        default=0.5,
        help="Fraction of consensus examples (default: 0.5)",
    )

    parser.add_argument(
        "--conflict",
        type=float,
        default=0.25,
        help="Fraction of conflict examples (default: 0.25)",
    )

    parser.add_argument(
        "--weakness",
        type=float,
        default=0.15,
        help="Fraction of weakness examples (default: 0.15)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_rag.jsonl",
        help="Output file path (default: data/synthetic_rag.jsonl)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Validate distribution
    total = args.consensus + args.conflict + args.weakness
    if total > 1.0:
        print("Error: Distribution fractions sum to > 1.0")
        return 1

    mixed_frac = 1.0 - total

    distribution = {
        'consensus': args.consensus,
        'conflict': args.conflict,
        'weakness': args.weakness,
        'mixed': mixed_frac,
    }

    print(f"\n{'='*60}")
    print("SYNTHETIC DATASET GENERATION")
    print(f"{'='*60}\n")

    # Generate dataset
    examples = generate_dataset(args.n, distribution)

    # Save to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\n✓ Dataset saved to: {output_path}")
    print(f"  Total examples: {len(examples)}")
    print(f"  Hallucination rate: {sum(ex['label'] for ex in examples) / len(examples):.1%}")

    return 0


if __name__ == "__main__":
    exit(main())
