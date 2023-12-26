## Geometry

$N$ disjoint spheres in $\mathbb{R}^3$:
- **Position vectors of the sphere centers**: $\mathbf{p_j} \in \mathbb{R}^3$, with $j \in \{1,...,N\}$.  
- **Radii**: $\tilde{r}_j\in \mathbb{R}^+$, with $j \in \{1,...,N\}$.
- **Interior of sphere $j$**: $\Omega_j:=\{ \mathbf{x} \in \mathbb{R}^3:||\mathbf{x}-\mathbf{p_j}||_2<\tilde{r}_j \}$.
- **Boundary of sphere $j$**: $\Gamma_j:=\partial \Omega_j$.
- **Exterior medium, $\Omega_0$**: defined as $\mathbb{R}^3$ without the spheres and their boundaries.  

## Free space fundamental solution

### Laplace equation
The **free space** **fundamental solution** of the **Laplace equation**, which satisfies the **radiation condition** is:
$$g\left(\mathbf{r},\mathbf{r'}\right):= \frac{1}{4\pi ||\mathbf{r}-\mathbf{r}'||_2}, \quad \mbox{with } \mathbf{r}\not = \mathbf{r'}, \quad \mbox{and } -\Delta g \left(\mathbf{r},\mathbf{r'}\right)=\delta \left(\mathbf{r}-\mathbf{r'}\right).$$

### Helmholtz equation
The **free space** **fundamental solution** of the **Helmholtz equation** with $k>0$, which satisfies the **radiation condition** is:
$$g_j\left(\mathbf{r},\mathbf{r'}\right):= \frac{e^{ik_j||\mathbf{r}-\mathbf{r}'||}}{4\pi ||\mathbf{r}-\mathbf{r}'||}, \quad \mbox{with } \mathbf{r}\not = \mathbf{r'}, \quad \mbox{and } -\left( \Delta + k_j^2 \right) g_j \left(\mathbf{r},\mathbf{r'}\right)=\delta \left(\mathbf{r}-\mathbf{r'}\right).$$


## Layer operators

Single and double layer operators defined for smooth densities:
$$DL_{0j} \left(\psi\right)\left(\mathbf{r}\right):=\int_{\Gamma_j}  \psi\left(\mathbf{r}'\right) \nabla g_0\left(\mathbf{r},\mathbf{r'}\right) \cdot \widehat{\mathbf{n}}_{0j} \ dS',$$
$$SL_{0j} \left(\psi\right)\left(\mathbf{r}\right):=\int_{\Gamma_j}  {\psi\left(\mathbf{r}'\right) g_0\left(\mathbf{r},\mathbf{r'}\right) dS'},$$
$$DL_j \left(\psi\right)\left(\mathbf{r}\right):=\int_{\Gamma_j}  {\psi\left(\mathbf{r}'\right) \nabla g_j\left(\mathbf{r},\mathbf{r'}\right) \cdot \widehat{\mathbf{n}}_{j} \ dS'},$$
$$SL_j \left(\psi\right)\left(\mathbf{r}\right):=\int_{\Gamma_j}  {\psi\left(\mathbf{r}' \right) g_j\left(\mathbf{r},\mathbf{r'}\right) dS'}$$

with the gradient being taken with respect to $\mathbf{r}'$, $\widehat{\mathbf{n}}_j$ being the exterior normal vector of $\Omega_j$, and $\hat{\mathbf{n}}_j = -\hat{\mathbf{n}}_{0j}$.

These operators are linear and continuous in the following Sobolev spaces:
$$DL_{0j}: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc} \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right),$$
$$SL_{0j}: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc}  \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right)  , $$
$$DL_{j}: H^{\frac{1}{2}}(\Gamma_j)\rightarrow H^1_{loc} \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right),$$
$$SL_{j}: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc}\left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right).$$

## Laplace



## Helmholtz