# Algorithm 2: Quantum Annealing Learning Search for QUBO Problems

## Input
- Q: symmetric matrix defining the QUBO problem
- A: annealer adjacency matrix
- E(Θ, z): annealer energy function
- g(P, p): permutation modification function
- pδ: minimum permutation probability (0 < pδ < 0.5)
- η: probability decreasing rate
- q: candidate perturbation probability
- N: number of iterations at constant p
- λ₀: initial balancing factor
- k: number of annealer runs per iteration
- i_max, N_max, d_min: termination parameters

## Output
- z*: approximate solution minimizing f(z) = zᵀQz

---

## Initialization

1. Define:
   f_Q(z) = zᵀ Q z

2. Set:
   P ← Iₙ  
   p ← 1

3. Generate permutations:
   P₁ ← g(P, 1)  
   P₂ ← g(P, 1)

4. Initialize weights:
   Θ₁ ← P₁ᵀ Q P₁ ∘ A  
   Θ₂ ← P₂ᵀ Q P₂ ∘ A

5. Run annealer k times:
   z₁ ← P₁ᵀ argmin_z E(Θ₁, z)  
   z₂ ← P₂ᵀ argmin_z E(Θ₂, z)

6. Evaluate:
   f₁ ← f_Q(z₁)  
   f₂ ← f_Q(z₂)

7. Initialize best and candidate:
   if f₁ < f₂:
       z* ← z₁  
       f* ← f₁  
       P* ← P₁  
       z′ ← z₂
   else:
       z* ← z₂  
       f* ← f₂  
       P* ← P₂  
       z′ ← z₁

8. Initialize tabu matrix:
   if f₁ ≠ f₂:
       S ← z′ ⊗ z′ − Iₙ + diag(z′)
   else:
       S ← 0

9. Initialize:
   e ← 0  // ogni volta che la solzione rimane la stessa e +1 altrimento e si resetta a 0
   d ← 0  // conta quante soluzioni diverse ma non migliori trova
   i ← 0  
   λ ← λ₀

---

## Iterative Process

Repeat:

10. Update problem:
    Q′ ← Q + λS

11. Update permutation probability:
    if N divides i:
        p ← p − (p − pδ)η

12. Generate new permutation:
    P ← g(P*, p)

13. Update weights:
    Θ′ ← Pᵀ Q′ P ∘ A

14. Run annealer k times:
    z′ ← Pᵀ argmin_z E(Θ′, z)

15. With probability q:
    z′ ← h(z′, p) // fa qeusto perche non esplora tutto il dominio e questo porta ad aver troato solo un minimo locale non globale

16. If z′ ≠ z*:

17. Evaluate:
    f′ ← f_Q(z′)

18. If f′ < f*:
    - swap(z′, z*)  
    - f* ← f′  
    - P* ← P  
    - e ← 0  
    - d ← 0  
    - Update tabu:
      S ← S + z′ ⊗ z′ − Iₙ + diag(z′)

19. Else:
    - d ← d + 1  
    - With probability p^(f′ − f*):
        swap(z′, z*)  
        f* ← f′  
        P* ← P  
        e ← 0

20. Update λ:
    λ ≤ λ₀

Else:
    e ← e + 1

21. i ← i + 1

Until:
    i = i_max OR (e + d ≥ N_max AND d < d_min)

---

## Return

- z*

---

# Algorithm 3: Permutation Matrix Modification Function g(P, p)

## Input
- P: permutation matrix
- p: probability of modifying elements

## Output
- P′: modified permutation matrix

---

1. Initialize associative map m

2. For each i = 1 to n:
   - with probability p:
     m[i] ← i   (select element for shuffling)

3. Shuffle the map m

4. For each i = 1 to n:
   - if i ∈ m:
       p′ᵢ ← p_{m[i]}
   - else:
       p′ᵢ ← pᵢ

5. Return P′

---

# Algorithm 4: Candidate Perturbation Function h(z, p)

## Input
- z: vector in {−1, 1}ⁿ
- p: probability of flipping a component

## Output
- perturbed vector z

--- 

1. For each i = 1 to n:
   - with probability p:
       zᵢ ← −zᵢ

2. Return z