# Algorithm 1: Quantum Annealing Learning Search (General Scheme)

## Input
- Objective function: \( f(z) \), with \( z \in \{-1, 1\}^n \)
- Annealer adjacency matrix: \( A \)
- Energy function: \( E(\Theta, z) \)

## Output
- Approximate solution \( z^* \) minimizing \( f(z) \)

---

## Initialization

1. Generate two random encodings:
   - \( \mu_1[f](z) := E(\Theta_1, \pi_1(z)) \)
   - \( \mu_2[f](z) := E(\Theta_2, \pi_2(z)) \)

2. Run the annealer to obtain:
   - \( z_1 \approx \arg\min E(\Theta_1, \cdot) \)
   - \( z_2 \approx \arg\min E(\Theta_2, \cdot) \)

3. Evaluate (le due z sono già riportate allo spazio originale):
   - \( f(z_1), f(z_2)\)

4. Set:
   - Best solution \( z^* \) = better of \( z_1, z_2 \) // il mim tra f_z1 e f_z2
   - Current encoding \( \mu^*  // mem la mu corrente\)
   

5. Set:
   - \( z' \) = worse solution

6. Initialize tabu matrix:
   - \( S \leftarrow z' \otimes z' - I + \text{diag}(z') \)

7. Initialize balancing factor:
   - \( \lambda \)

---

## Iterative Process

Repeat:

8. Generate new encoding from \( \mu^* \) and \( S \)

9. Define:
   \[
   \mu[f](z) := E(\Theta[f] + \lambda S_\pi \circ A, \pi(z))
   \]
   where:
   \[
   S_\pi = P_\pi^T S P_\pi
   \]

10. Run the annealer to obtain:
   - \( z' \approx \arg\min E(\Theta, \cdot) \)

11. If \( z' \neq z^* \):

12. Evaluate:
   - \( f(z') \)

13. If \( f(z') < f(z^*) \):
   - Update best solution:
     - \( z^* \leftarrow z' \)
     - \( \mu^* \leftarrow \mu \)

14. Update tabu matrix:
   \[
   S \leftarrow S + z' \otimes z' - I + \text{diag}(z')
   \]

15. Update balancing factor:
   - \( \lambda \)

Until stopping condition is met

---

## Return

- \( z^* \)