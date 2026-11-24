#!/usr/bin/env python3
"""
Bayesian Network solver for Assignment 2.1: Compute Blood Types.

Models ABO blood type inheritance and test results as a Bayesian network
and performs exact inference via Variable Elimination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allele prior probabilities by country: (A, B, O)
ALLELES = ("A", "B", "O")
BLOOD_TYPES = ("O", "A", "B", "AB")

# 6 unordered diploid genotypes
GENOTYPES = ("AA", "AO", "BB", "BO", "AB", "OO")
GENOTYPE_TO_IDX = {g: i for i, g in enumerate(GENOTYPES)}
BLOOD_TYPE_TO_IDX = {bt: i for i, bt in enumerate(BLOOD_TYPES)}

COUNTRY_ALLELE_PRIORS: Dict[str, Dict[str, float]] = {
    "North Wumponia": {"A": 0.50, "B": 0.25, "O": 0.25},
    "South Wumponia": {"A": 0.15, "B": 0.60, "O": 0.25},
}

# Mapping from genotype to blood type
GENOTYPE_TO_BLOOD_TYPE: Dict[str, str] = {
    "AA": "A",
    "AO": "A",
    "BB": "B",
    "BO": "B",
    "AB": "AB",
    "OO": "O",
}


def canonical_genotype(a1: str, a2: str) -> str:
    """Return canonical unordered genotype string for two alleles."""
    pair = {a1, a2}
    if pair == {"A"}:
        return "AA"
    if pair == {"B"}:
        return "BB"
    if pair == {"O"}:
        return "OO"
    if pair == {"A", "O"}:
        return "AO"
    if pair == {"B", "O"}:
        return "BO"
    if pair == {"A", "B"}:
        return "AB"
    raise ValueError(f"Unknown allele pair: {a1}, {a2}")


def genotype_prior(country: str) -> np.ndarray:
    """Prior distribution over the 6 unordered genotypes for a country."""
    if country not in COUNTRY_ALLELE_PRIORS:
        raise ValueError(f"Unknown country: {country}")
    p = COUNTRY_ALLELE_PRIORS[country]
    dist = np.zeros(len(GENOTYPES), dtype=np.float64)
    for a1 in ALLELES:
        for a2 in ALLELES:
            g = canonical_genotype(a1, a2)
            dist[GENOTYPE_TO_IDX[g]] += p[a1] * p[a2]
    return dist


def genotype_alleles(g: str) -> Tuple[str, str]:
    """Return the two alleles of a genotype."""
    return g[0], g[1]


def build_two_parent_cpt() -> np.ndarray:
    """
    CPT P(child_genotype | father_genotype, mother_genotype)
    Shape: (6, 6, 6) indexed by (child, father, mother).
    """
    cpt = np.zeros((6, 6, 6), dtype=np.float64)
    for fi, fg in enumerate(GENOTYPES):
        fa1, fa2 = genotype_alleles(fg)
        for mi, mg in enumerate(GENOTYPES):
            ma1, ma2 = genotype_alleles(mg)
            for fa in (fa1, fa2):
                for ma in (ma1, ma2):
                    cg = canonical_genotype(fa, ma)
                    cpt[GENOTYPE_TO_IDX[cg], fi, mi] += 0.25
    return cpt


def build_one_parent_cpt(country: str) -> np.ndarray:
    """
    CPT P(child_genotype | known_parent_genotype) when the other parent is
    unknown (drawn from the country's allele prior).
    Shape: (6, 6) indexed by (child, parent).
    """
    p = COUNTRY_ALLELE_PRIORS[country]
    cpt = np.zeros((6, 6), dtype=np.float64)
    for pi, pg in enumerate(GENOTYPES):
        pa1, pa2 = genotype_alleles(pg)
        for pa in (pa1, pa2):
            for ua in ALLELES:
                cg = canonical_genotype(pa, ua)
                cpt[GENOTYPE_TO_IDX[cg], pi] += 0.5 * p[ua]
    return cpt


def mix_blood_types(bt1: str, bt2: str) -> str:
    """Return the blood type of a mixture of two blood samples."""
    has_a = ("A" in bt1) or ("A" in bt2)
    has_b = ("B" in bt1) or ("B" in bt2)
    if has_a and has_b:
        return "AB"
    if has_a:
        return "A"
    if has_b:
        return "B"
    return "O"


class Factor:
    """A discrete factor over a list of named variables."""

    def __init__(self, variables: Sequence[str], values: np.ndarray):
        self.variables: List[str] = list(variables)
        self.values: np.ndarray = np.asarray(values, dtype=np.float64)
        if len(self.variables) != self.values.ndim:
            raise ValueError(
                f"Variables {self.variables} do not match array ndim {self.values.ndim}"
            )

    def multiply(self, other: "Factor") -> "Factor":
        """Multiply two factors over their union of variables."""
        if not self.variables:
            return Factor(other.variables, self.values * other.values)
        if not other.variables:
            return Factor(self.variables, self.values * other.values)

        new_vars = list(self.variables)
        for v in other.variables:
            if v not in new_vars:
                new_vars.append(v)

        # Align axes of self and other to new_vars
        self_shape = [
            self.values.shape[self.variables.index(v)] if v in self.variables else 1
            for v in new_vars
        ]
        other_perm = [other.variables.index(v) for v in new_vars if v in other.variables]
        other_transposed = np.transpose(other.values, other_perm)
        other_shape = [
            other.values.shape[other.variables.index(v)] if v in other.variables else 1
            for v in new_vars
        ]

        result = self.values.reshape(self_shape) * other_transposed.reshape(other_shape)
        return Factor(new_vars, result)

    def sum_out(self, var: str) -> "Factor":
        """Marginalize (sum out) a variable from this factor."""
        axis = self.variables.index(var)
        new_vars = [v for v in self.variables if v != var]
        new_vals = np.sum(self.values, axis=axis)
        return Factor(new_vars, new_vals)


def variable_elimination(factors: List[Factor], query_var: str) -> np.ndarray:
    """
    Compute the normalized posterior distribution over query_var
    using Variable Elimination with a min-fill / min-degree heuristic.
    """
    working = list(factors)
    all_vars = set()
    for f in working:
        all_vars.update(f.variables)

    elim_vars = [v for v in all_vars if v != query_var]

    while elim_vars:
        # Choose variable that minimizes the size of the resulting product factor
        best_var = None
        best_score = None
        for v in elim_vars:
            scope = set()
            for f in working:
                if v in f.variables:
                    scope.update(f.variables)
            score = len(scope)
            if best_score is None or score < best_score:
                best_score = score
                best_var = v

        assert best_var is not None
        elim_vars.remove(best_var)

        with_var = [f for f in working if best_var in f.variables]
        without_var = [f for f in working if best_var not in f.variables]

        prod = with_var[0]
        for f in with_var[1:]:
            prod = prod.multiply(f)
        reduced = prod.sum_out(best_var)
        without_var.append(reduced)
        working = without_var

    # Multiply remaining factors (only query_var and scalars remain)
    result = working[0]
    for f in working[1:]:
        result = result.multiply(f)

    # Ensure query_var is the only variable
    if result.variables != [query_var]:
        raise RuntimeError(f"Unexpected remaining variables: {result.variables}")

    total = float(np.sum(result.values))
    if total == 0.0:
        raise ValueError("Zero probability evidence encountered.")
    return result.values / total


def build_factors(problem: dict) -> List[Factor]:
    """Construct the Bayesian network factors for a problem instance."""
    country: str = problem["country"]
    family_tree: List[dict] = problem["family-tree"]
    test_results: List[dict] = problem["test-results"]
    queries: List[dict] = problem["queries"]

    # Collect all people and their father/mother relations
    people = set()
    fathers: Dict[str, str] = {}
    mothers: Dict[str, str] = {}

    for rel in family_tree:
        subj = rel["subject"]
        obj = rel["object"]
        rtype = rel["relation"]
        people.add(subj)
        people.add(obj)
        if rtype == "father-of":
            fathers[obj] = subj
        elif rtype == "mother-of":
            mothers[obj] = subj
        else:
            raise ValueError(f"Unknown relation: {rtype}")

    for test in test_results:
        if "person" in test:
            people.add(test["person"])
        if "person-1" in test:
            people.add(test["person-1"])
        if "person-2" in test:
            people.add(test["person-2"])

    for q in queries:
        people.add(q["person"])

    prior = genotype_prior(country)
    two_parent_cpt = build_two_parent_cpt()
    one_parent_cpt = build_one_parent_cpt(country)

    factors: List[Factor] = []

    # 1. Genotype prior / inheritance CPTs for each person
    for person in sorted(people):
        father = fathers.get(person)
        mother = mothers.get(person)
        if father is not None and mother is not None:
            factors.append(Factor([person, father, mother], two_parent_cpt))
        elif father is not None:
            factors.append(Factor([person, father], one_parent_cpt))
        elif mother is not None:
            factors.append(Factor([person, mother], one_parent_cpt))
        else:
            factors.append(Factor([person], prior))

    # 2. Evidence factors from test results
    for test in test_results:
        ttype = test["type"]
        if ttype == "bloodtype-test":
            person = test["person"]
            observed = test["result"]
            likelihood = np.array(
                [1.0 if GENOTYPE_TO_BLOOD_TYPE[g] == observed else 0.0 for g in GENOTYPES],
                dtype=np.float64,
            )
            factors.append(Factor([person], likelihood))

        elif ttype == "mixed-bloodtype-test":
            p1 = test["person-1"]
            p2 = test["person-2"]
            observed = test["result"]
            if p1 == p2:
                likelihood = np.array(
                    [1.0 if GENOTYPE_TO_BLOOD_TYPE[g] == observed else 0.0 for g in GENOTYPES],
                    dtype=np.float64,
                )
                factors.append(Factor([p1], likelihood))
            else:
                likelihood = np.zeros((6, 6), dtype=np.float64)
                for i, g1 in enumerate(GENOTYPES):
                    bt1 = GENOTYPE_TO_BLOOD_TYPE[g1]
                    for j, g2 in enumerate(GENOTYPES):
                        bt2 = GENOTYPE_TO_BLOOD_TYPE[g2]
                        if mix_blood_types(bt1, bt2) == observed:
                            likelihood[i, j] = 1.0
                factors.append(Factor([p1, p2], likelihood))

        elif ttype == "pair-bloodtype-test":
            p1 = test["person-1"]
            p2 = test["person-2"]
            r1 = test["result-1"]
            r2 = test["result-2"]
            # 75% chance: (p1 -> r1, p2 -> r2); 25% chance: (p1 -> r2, p2 -> r1)
            if p1 == p2:
                likelihood = np.zeros(6, dtype=np.float64)
                for i, g in enumerate(GENOTYPES):
                    bt = GENOTYPE_TO_BLOOD_TYPE[g]
                    if bt == r1 and bt == r2:
                        likelihood[i] = 1.0
                factors.append(Factor([p1], likelihood))
            else:
                likelihood = np.zeros((6, 6), dtype=np.float64)
                for i, g1 in enumerate(GENOTYPES):
                    bt1 = GENOTYPE_TO_BLOOD_TYPE[g1]
                    for j, g2 in enumerate(GENOTYPES):
                        bt2 = GENOTYPE_TO_BLOOD_TYPE[g2]
                        prob = 0.0
                        if bt1 == r1 and bt2 == r2:
                            prob += 0.75
                        if bt1 == r2 and bt2 == r1:
                            prob += 0.25
                        likelihood[i, j] = prob
                factors.append(Factor([p1, p2], likelihood))
        else:
            raise ValueError(f"Unknown test type: {ttype}")

    return factors


def solve_problem(problem: dict) -> List[dict]:
    """Solve a single problem dictionary and return the list of query answers."""
    factors = build_factors(problem)
    solutions: List[dict] = []

    for query in problem["queries"]:
        qtype = query["type"]
        if qtype != "bloodtype":
            raise ValueError(f"Unknown query type: {qtype}")
        person = query["person"]
        genotype_dist = variable_elimination(factors, person)

        bt_dist = {bt: 0.0 for bt in BLOOD_TYPES}
        for i, g in enumerate(GENOTYPES):
            bt_dist[GENOTYPE_TO_BLOOD_TYPE[g]] += float(genotype_dist[i])

        solutions.append(
            {
                "type": "bloodtype",
                "person": person,
                "distribution": bt_dist,
            }
        )

    return solutions


def solve_directory(input_dir: Path, output_dir: Path) -> int:
    """Solve all problem-*.json files in input_dir and write solution-*.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_files = sorted(input_dir.glob("problem-*.json"))
    for problem_path in problem_files:
        with open(problem_path, "r", encoding="utf-8") as fp:
            problem = json.load(fp)
        solution = solve_problem(problem)
        sol_name = problem_path.name.replace("problem-", "solution-", 1)
        sol_path = output_dir / sol_name
        with open(sol_path, "w", encoding="utf-8") as fp:
            json.dump(solution, fp, indent=4)
            fp.write("\n")
    return len(problem_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve ABO blood type Bayesian network problems."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a single problem JSON file or a directory of problem-*.json files.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to output JSON file (for single file) or output directory (for directory).",
    )
    args = parser.parse_args()

    if args.input_path.is_dir():
        count = solve_directory(args.input_path, args.output_path)
        print(f"Solved {count} problems into {args.output_path}")
    elif args.input_path.is_file():
        with open(args.input_path, "r", encoding="utf-8") as fp:
            problem = json.load(fp)
        solution = solve_problem(problem)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as fp:
            json.dump(solution, fp, indent=4)
            fp.write("\n")
        print(f"Solved {args.input_path} -> {args.output_path}")
    else:
        raise FileNotFoundError(f"Input path not found: {args.input_path}")


if __name__ == "__main__":
    main()
