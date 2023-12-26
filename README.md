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

with the gradient being taken with respect to $\mathbf{r}'$, $\widehat{\mathbf{n}}_j$ being the exterior normal vector of $\Omega_j$, and $\widehat{\mathbf{n}}_j = -\widehat{\mathbf{n}}_{0j}$.

These operators are linear and continuous in the following Sobolev spaces:
$$DL_{0j}: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc} \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right),$$
$$SL_{0j}: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc}  \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right)  , $$
$$DL_{j}: H^{\frac{1}{2}}(\Gamma_j)\rightarrow H^1_{loc} \left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right),$$
$$SL_{j}: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^1_{loc}\left(\mathbb{R}^3 \setminus \cup_{j=1}^{N}\Gamma_j\right).$$

## Boundary integral operators

For $u\in C^\infty(\overline\Omega)$, Dirichlet and Neumann traces operators are defined as 
$$\trazaD u := u|_{\Gamma},\qquad \trazaN u := \nabla  u|_{\Gamma} \cdot \Unormal,$$
where $\Unormal$ is the exterior unit normal.

$$\begin{align}
	V_{i,j}^0 &:=  \frac{1}{2} \left( \trazaD^{i} SL_{0j} + \trazaD^{0i} SL_{0j} \right) ,
	& V_{j}&:= \frac{1}{2} \left(  \trazaD^{0j} SL_{j} + \trazaD^{j} SL_{j} \right) ,\nonumber \\
	K_{i,j}^0&:= \frac{1}{2} \left(\trazaD^{i} DL_{0j} + \trazaD^{0i} DL_{0j} \right) ,
	&K_{j}&:= \frac{1}{2} \left(\trazaD^{0j} DL_{j} + \trazaD^{j} DL_{j} \right), \label{BIOS-definition}\\
	K^{*0}_{i,j}&:= \frac{1}{2} \left( - \trazaN ^{i} SL_{0j} + \trazaN ^{0i} SL_{0j}  \right),
	 & K^{*}_{j} &:= \frac{1}{2} \left( -\trazaN ^{0j} SL_{j}  + \trazaN ^{j} SL_{j} \right), \nonumber \\
	W_{i,j}^0 &:= -\frac{1}{2} \left( - \trazaN^{i} DL_{0j}  + \trazaN^{0i} DL_{0j} \right) ,
	& W_{j} &:=- \frac{1}{2} \left( -\trazaN^{0j} DL_{j} + \trazaN^{j} DL_{j} \right). \nonumber
\end{align}$$

These operators are linear and continuous in the following Sobolev spaces: 
$$\begin{align*}
	V_{{i},j}^0 &: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^{\frac{1}{2}}(\Gamma_i),
	&V_{j}&: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^{\frac{1}{2}}(\Gamma_j),\\
	W_{{i},j}^0&: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^{-\frac{1}{2}}(\Gamma_i),
	&W_{j}&: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^{-\frac{1}{2}}(\Gamma_j) ,\\
	K_{{i},j}^0&: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^{\frac{1}{2}}(\Gamma_i),
	&K_{j}&: H^{\frac{1}{2}}(\Gamma_j) \rightarrow H^{\frac{1}{2}}(\Gamma_j) ,\\
	K^{*0}_{{i},j}&: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^{-\frac{1}{2}}(\Gamma_i),
	&K^*_{j}&: H^{-\frac{1}{2}}(\Gamma_j) \rightarrow H^{-\frac{1}{2}}(\Gamma_j).
\end{align*}$$

Since the domains are smooth, the jump relations for the potentials across a closed boundary yield
$$\begin{align*}
	V_{{i},j}^0 &=   \trazaD^{0{i}} SL_{0j},
	& V_{j}&=  \trazaD^{j} SL_{j},\\
	W_{{i},j}^0 &=-  \trazaN^{0{i}} DL_{0j},
	& W_{j} &=- \trazaN^{j} DL_{j}, \\
	  K_{{i},j}^0&= \trazaD^{0{i}} DL_{0j}\mbox{ with } {i} \not=j,
	 & K^{*0}_{{i},j} &= \trazaN ^{0{i}} SL_{0j}\mbox{ with } {i}  \not=j,
\end{align*}$$
and
$$\begin{align*}
	K_{j,j}^0(\psi) &= \frac{1}{2}\psi +\trazaD^{0j} {DL_{0j}(\psi)} ,
	&K_{j}(\psi) &= \frac{1}{2} \psi +\trazaD^{j} {DL_{j}(\psi)} ,\\
	 K^{*0}_{j,j}(\psi) &= -\frac{1}{2} \psi + \trazaN^{0j} {SL_{0j}(\psi)},
	&K^*_{j}(\psi) &= -\frac{1}{2} \psi + \trazaN^j {SL_{j}(\psi)}.
\end{align*}$$