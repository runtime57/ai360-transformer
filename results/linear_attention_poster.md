## 1. What changes compared with softmax attention?

For a single attention head, the output at position $t$ is

$$
y_t =
\frac{\sum_{j=1}^{t}s(q_t,k_j)\,v_j}
{\sum_{j=1}^{t}s(q_t,k_j)}.
$$

The two attention variants use different similarity functions:

$$
\begin{aligned}
s_{\mathrm{softmax}}(q,k)
&=\exp\left(\frac{q^\top k}{\sqrt d}\right),\\
s_{\mathrm{linear}}(q,k)
&=\phi(q)^\top\phi(k),
\qquad \phi(x)=\mathrm{ELU}(x)+1.
\end{aligned}
$$

## 2. Causal Masking

Define the accumulated key-value state and normalization state:

$$
S_t=\sum_{j=1}^{t}v_j\phi(k_j)^\top,
\qquad
z_t=\sum_{j=1}^{t}\phi(k_j).
$$

Rearranging the sums gives

$$
\boxed{
y_t=
\frac{S_t\phi(q_t)}
{\max\left(\phi(q_t)^\top z_t,\varepsilon\right)}
}
$$

The states can be updated incrementally:

$$
\boxed{
S_t=S_{t-1}+v_t\phi(k_t)^\top,
\qquad
z_t=z_{t-1}+\phi(k_t)
}
$$

with $S_0=0$ and $z_0=0$. The small lower bound $\varepsilon$ stabilizes the division.


## 3. Computational complexity

| Attention core | Computational complexity |
| --- | ---: |
| Softmax attention | $O(N^2d)$ |
| Linear attention | $O(Nd^2)$ |


<!-- можно скипнуть 4 -->
## 4. Сombining linear attention with RoPE

Apply the position-dependent rotations after the feature map:

$$
\tilde q_t=R_t\phi(q_t),
\qquad
\tilde k_j=R_j\phi(k_j),
$$

where $R_t$ is the RoPE rotation matrix at position $t$. Then

$$
y_t^{\mathrm{RoPE}}=
\frac{\sum_{j\le t}(\tilde q_t^\top\tilde k_j)\,v_j}
{\max\left(\phi(q_t)^\top\sum_{j\le t}\phi(k_j),\varepsilon\right)}.
$$

## 5. Сonclusion

> In our TinyStories experiments, linear attention achieves slightly worse language-modeling quality than softmax attention. In the configuration with a context length of 8192, it reduces mean training time per batch and increases generation throughput. With a context length of 512, our implementation of linear attention is slower than softmax attention. Linear asymptotic complexity therefore does not guarantee faster execution for short contexts.
