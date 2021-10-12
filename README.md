# All codes related to the paper of momenta quantization of eigen modes


### My VSH

The idea is to define complex VSH as
```math
\psi_{mn} (\mathbf{r}) = z_n(r) Y^m_n(\theta, \varphi),
```
and then
```math
\mathbf{M}_{m n} = \nabla \times (\mathbf{r} \psi_{m n}), \qquad \mathbf{N}_{m n} = \frac{1}{k} \nabla \times \mathbf{M}_{m n}, \qquad \mathbf{L}_{m n} = \nabla \psi_{m n}
```
then the result is symmetryc with respect to $`m \to -m`$.

We get in the basis $``(\mathbf{\hat{r}}, \mathbf{\hat{\theta}}, \mathbf{\hat{\varphi}})``$:
```math
\mathbf{M}_{mn} = \begin{bmatrix}
0 \\
\frac{im}{ \sin \theta} z_n(r) Y^m_n(\theta, \varphi) \\
-z_n(r) \frac{d}{d\theta} Y^m_n(\theta, \varphi)
\end{bmatrix}, \qquad
\mathbf{N}_{mn} = \begin{bmatrix}
n (n + 1) \frac{z_n(r)}{r} Y^m_n(\theta, \varphi) \\
\frac{1}{r} \frac{d }{dr} \left[ r z_n(r) \right]  \frac{d}{d\theta} Y^m_n(\theta, \varphi) \\
\frac{im}{\sin \theta} \frac{1}{r} \frac{d }{dr} \left[ r z_n(r) \right] Y^m_n(\theta, \varphi)
\end{bmatrix}, \qquad
\mathbf{L}_{mn} = \begin{bmatrix}
 z_n^\prime(r) Y^m_n(\theta, \varphi) \\
 \frac{z_n(r)}{r} \frac{d}{d \theta} Y^m_n(\theta, \varphi)\\
\frac{i m }{\sin \theta} \frac{z_n}{r} Y^m_n(\theta, \varphi)
\end{bmatrix}
```
