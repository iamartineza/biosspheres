# Installing instructions

An example installation using conda:
- Clone the repository.
- Create a new conda environment and activate it.
- Add conda-forge channel
`conda config --add channels conda-forge`
- cd to local directory.
- Install the packages in requirements.txt using
`conda install --file requirements.txt`
- If the jupyter notebooks are needed install also
`conda install --file requirements_notebooks.txt`
- Install pip (if not installed)
`conda install pip`
- Install biosspheres
`pip install --editable .`

# Overview of the library

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
```math
DL_{0j} \left(\psi\right)\left(\mathbf{r}\right):=\int_{\Gamma_j}  \psi\left(\mathbf{r}'\right) \nabla g_0\left(\mathbf{r},\mathbf{r'}\right) \cdot \widehat{\mathbf{n}}_{0j} \ dS',\\
```
```math
SL_{0j} \left(\psi\right)\left(\mathbf{r}\right):=\int_{\Gamma_j}  {\psi\left(\mathbf{r}'\right) g_0\left(\mathbf{r},\mathbf{r'}\right) dS'},
```
```math
DL_j \left(\psi\right)\left(\mathbf{r}\right):=\int_{\Gamma_j}  {\psi\left(\mathbf{r}'\right) \nabla g_j\left(\mathbf{r},\mathbf{r'}\right) \cdot \widehat{\mathbf{n}}_{j} \ dS'},
```
```math
SL_j \left(\psi\right)\left(\mathbf{r}\right):=\int_{\Gamma_j}  {\psi\left(\mathbf{r}' \right) g_j\left(\mathbf{r},\mathbf{r'}\right) dS'}
```

with the gradient being taken with respect to $\mathbf{r}'$, $\widehat{\mathbf{n}}_j$ being the exterior normal vector of $\Omega_j$, and $\widehat{\mathbf{n}}_j=-\widehat{\mathbf{n}}_{0j}$.

These operators are linear and continuous in the following Sobolev spaces:
$$DL_{0j}: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc} \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right),$$
$$SL_{0j}: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc}  \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right)  , $$
$$DL_{j}: H^{\frac{1}{2}}(\Gamma_j)\rightarrow H^1_{loc} \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right),$$
$$SL_{j}: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc}\left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right).$$

## Boundary integral operators

For $u\in C^\infty(\overline\Omega)$, Dirichlet and Neumann traces operators are defined as 
```math
\gamma_d u := u|_{\Gamma},\qquad \gamma_n u := \nabla  u|_{\Gamma} \cdot \widehat{n},
```
where $\widehat{n}$ is the exterior unit normal.

By density arguments, the definition of Dirichlet and Neumann traces operator can be extended to $u_j \in H^1_{loc}(\Omega_j)$, with $j \in \{ 0, ..., N \}$. We extend the notation as follows

```math
\gamma_d^{0j} u_0 := u_0|_{\Gamma_j},
```
```math
\gamma_d^{j} u_j := u_0|_{\Gamma_j},
```
```math
\gamma_n^{0j} u_0 := \nabla u_0|_{\Gamma_j} \cdot \widehat{n}_{0j},
```
```math
\gamma_n^{j} u_j := \nabla u_0|_{\Gamma_j}\cdot \widehat{n}_{j},
```
where $\widehat{n}_{j}$ is the exterior normal of $\Omega_j$, with $j\in \{ 1, ..., N\}$ and $\widehat{n}_{0j}=-\widehat{n}_{j}$.

Now, we recall the definition of the boundary integral operators:
```math
\begin{align*}
	V_{i,j}^0 &:=  \frac{1}{2} \left( \gamma_d^{i} SL_{0j} + \gamma_d^{0i} SL_{0j} \right) ,
	& V_{j}&:= \frac{1}{2} \left(  \gamma_d^{0j} SL_{j} + \gamma_d^{j} SL_{j} \right) ,\\
	K_{i,j}^0&:= \frac{1}{2} \left(\gamma_d^{i} DL_{0j} + \gamma_d^{0i} DL_{0j} \right) ,
	&K_{j}&:= \frac{1}{2} \left(\gamma_d^{0j} DL_{j} + \gamma_d^{j} DL_{j} \right),\\
	K^{*0}_{i,j}&:= \frac{1}{2} \left( - \gamma_n ^{i} SL_{0j} + \gamma_n ^{0i} SL_{0j}  \right),
	 & K^{*}_{j} &:= \frac{1}{2} \left( -\gamma_n ^{0j} SL_{j}  + \gamma_n ^{j} SL_{j} \right), \\
	W_{i,j}^0 &:= -\frac{1}{2} \left( - \gamma_n^{i} DL_{0j}  + \gamma_n^{0i} DL_{0j} \right) ,
	& W_{j} &:=- \frac{1}{2} \left( -\gamma_n^{0j} DL_{j} + \gamma_n^{j} DL_{j} \right).
\end{align*}
```

These operators are linear and continuous in the following Sobolev spaces: 
```math
\begin{align*}
	V_{{i},j}^0 &: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^{\frac{1}{2}}(\Gamma_i),
	&V_{j}&: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^{\frac{1}{2}}(\Gamma_j),\\
	W_{{i},j}^0&: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^{-\frac{1}{2}}(\Gamma_i),
	&W_{j}&: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^{-\frac{1}{2}}(\Gamma_j) ,\\
	K_{{i},j}^0&: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^{\frac{1}{2}}(\Gamma_i),
	&K_{j}&: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^{\frac{1}{2}}(\Gamma_j) ,\\
	K^{*0}_{{i},j}&: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^{-\frac{1}{2}}(\Gamma_i),
	&K^*_{j}&: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^{-\frac{1}{2}}(\Gamma_j).
\end{align*}
```

Since the domains are smooth, the jump relations for the potentials across a closed boundary yield
```math
\begin{align*}
	V_{{i},j}^0 &=   \gamma_d^{0{i}} SL_{0j},
	& V_{j}&=  \gamma_d^{j} SL_{j},\\
	W_{{i},j}^0 &=-  \gamma_n^{0{i}} DL_{0j},
	& W_{j} &=- \gamma_n^{j} DL_{j}, \\
	  K_{{i},j}^0&= \gamma_d^{0{i}} DL_{0j}\mbox{ with } {i} \not=j,
	 & K^{*0}_{{i},j} &= \gamma_n ^{0{i}} SL_{0j}\mbox{ with } {i}  \not=j,
\end{align*}
```
and
```math
\begin{align*}
	K_{j,j}^0(\psi) &= \frac{1}{2}\psi +\gamma_d^{0j} {DL_{0j}(\psi)} ,
	&K_{j}(\psi) &= \frac{1}{2} \psi +\gamma_d^{j} {DL_{j}(\psi)} ,\\
	 K^{*0}_{j,j}(\psi) &= -\frac{1}{2} \psi + \gamma_n^{0j} {SL_{0j}(\psi)},
	&K^*_{j}(\psi) &= -\frac{1}{2} \psi + \gamma_n^j {SL_{j}(\psi)}.
\end{align*}
```

## Comments about the code.

All Legendre's functions are computed using the package pyshtools 
([documentation of pyshtools](https://shtools.github.io/SHTOOLS/index.html))
