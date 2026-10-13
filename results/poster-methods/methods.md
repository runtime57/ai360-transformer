## RoPE

**Sin/cos:** позиционный вектор прибавляется к эмбеддингу токена перед проекциями Q, K, V:

$$
\mathrm{PE}_{m,2i}=\sin(m\theta_i),\qquad
\mathrm{PE}_{m,2i+1}=\cos(m\theta_i),\qquad
\theta_i=10000^{-2i/d_{\mathrm{model}}}.
$$

**RoPE:** вместо прибавления PE поворачивает пары координат Q и K внутри каждой головы. V не меняется; обучаемых параметров нет.

$$
\theta_i=10000^{-2i/d_h},\qquad
R(\varphi)=\begin{pmatrix}\cos\varphi&-\sin\varphi\\\sin\varphi&\cos\varphi\end{pmatrix},
$$

$$
q'_{m,[2i:2i+2]}=R(m\theta_i)q_{m,[2i:2i+2]},\qquad
k'_{n,[2i:2i+2]}=R(n\theta_i)k_{n,[2i:2i+2]}.
$$

Полная матрица поворота при $n=d_h$ (чётное):

$$
R_m=
\begin{pmatrix}
\cos(m\theta_0)&-\sin(m\theta_0)&\cdots&0&0\\
\sin(m\theta_0)&\cos(m\theta_0)&\cdots&0&0\\
\vdots&\vdots&\ddots&\vdots&\vdots\\
0&0&\cdots&\cos(m\theta_{n/2-1})&-\sin(m\theta_{n/2-1})\\
0&0&\cdots&\sin(m\theta_{n/2-1})&\cos(m\theta_{n/2-1})
\end{pmatrix}
\in\mathbb{R}^{n\times n}.
$$

Позиционный вклад в скалярное произведение определяется относительным смещением:

$$
(q'_m)^\top k'_n=q_m^\top R_{n-m}k_n.
$$

Здесь $d_h$ — размер головы, $R_m$ — поворот всех пар координат для позиции $m$.

## Gated attention

**Базовый attention:** выходы голов объединяются и проходят выходную проекцию.

$$
A_h=\operatorname{softmax}\!\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}+M\right)V_h,\qquad
y=W_O\operatorname{Concat}_h(A_h).
$$

**Gated attention:** добавляет обучаемый sigmoid-gate после attention, перед выходной проекцией. Gate зависит от входа и регулирует передачу информации:

$$
g=\sigma(W_gx+b_g),\qquad
y=W_O\operatorname{Concat}_h(A_h\odot g_h).
$$

Здесь $x$ — вход attention после нормализации, $M$ — причинная маска. $g\approx0$ подавляет выход; $g\approx1$ сохраняет его.

- **Headwise:** $g_h\in(0,1)$ — один коэффициент на голову и токен, общий для всех её координат.
- **Elementwise:** $g_h\in(0,1)^{d_h}$ — отдельный коэффициент на каждую координату головы и токен; более точное управление, больше параметров gate.
