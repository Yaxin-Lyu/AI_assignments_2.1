# Solution Summary: Assignment 2.1 – Compute Blood Types

## 1. Approach and Bayesian Network Modeling

I modeled the ABO blood type inheritance problem as a Bayesian Network and implemented exact probabilistic inference using **Variable Elimination (VE)**.

Each family member is represented by a discrete random variable with six possible unordered genotypes: {AA, AO, BB, BO, AB, OO}. I chose genotypes rather than blood types because inheritance operates at the allele level, and the same blood type can correspond to different genotypes with different inheritance probabilities.

The probability model handles three cases:

- **No known parents:** Genotype priors are computed from country-specific allele frequencies, assuming independent alleles (e.g., \(P(AO)=2p_Ap_O\)).
- **Both parents known:** A \(6\times6\times6\) conditional probability table (CPT) models the child inheriting one allele uniformly from each parent.
- **One parent known:** The missing parent's transmitted allele is marginalized using the country-specific allele distribution, resulting in a \(6\times6\) CPT. This avoids introducing unnecessary variables for unspecified parents.

## 2. Modeling Test Results

All test results are represented as likelihood factors over the relevant genotype variables:

- **Bloodtype test:** A deterministic factor assigns probability 1 to genotypes matching the observed blood type and 0 otherwise.
- **Mixed-bloodtype test:** A binary factor checks whether the combined A/B antigens of two blood samples match the observed result.
- **Pair-bloodtype test:** A binary likelihood factor models the 25% probability of swapped samples:

\[
P(r_1,r_2\mid G_1,G_2)
=0.75\,\mathbb{I}[bt_1=r_1\land bt_2=r_2]
+0.25\,\mathbb{I}[bt_1=r_2\land bt_2=r_1]
\]

Representing observations as factors allows all three test types to be processed by the same inference algorithm without introducing separate inference procedures.

## 3. Inference Algorithm and Elimination Order

I chose **Variable Elimination** because the assignment requires complete posterior distributions rather than only the most probable blood type. Compared with enumerating all \(6^n\) genotype assignments, VE exploits the factorized structure of the Bayesian Network.

For each query, the algorithm:

1. Selects a non-query variable using a greedy **minimum-scope-size heuristic**.
2. Multiplies all factors containing that variable and sums it out.
3. Multiplies the remaining factors, normalizes the resulting genotype distribution, and aggregates the six genotypes into four blood types.

The elimination order is an important design decision because it determines the size of intermediate factors. My heuristic selects the variable whose associated factors have the smallest combined variable scope. This is computationally inexpensive and aims to limit intermediate tensor sizes, although it does not guarantee the optimal elimination order.

## 4. Verification and Limitations

The implementation uses exact probabilistic inference and supports all three test types and both country-specific prior distributions.

**Verification:**
- Passed all 80 example problems using `solution_check.py`, with tolerance \(\epsilon=10^{-6}\).
- Generated solution files for all 80 assignment problems.

The main limitation is that inference complexity depends on the family network structure and elimination order. Highly connected networks may produce large intermediate factors. More advanced heuristics, such as Min-Fill or Weighted Min-Fill, could improve performance. Reusing intermediate inference results across multiple queries could also reduce redundant computation.