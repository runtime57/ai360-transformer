## 1. Что меняется по сравнению с softmax attention?

Для одной головы внимания выход в позиции $t$ имеет вид

$$
y_t =
\frac{\sum_{j=1}^{t}s(q_t,k_j)\,v_j}
{\sum_{j=1}^{t}s(q_t,k_j)}.
$$

Два варианта внимания используют разные функции сходства:

$$
\begin{aligned}
s_{\mathrm{softmax}}(q,k)
&=\exp\left(\frac{q^\top k}{\sqrt d}\right)\\
s_{\mathrm{linear}}(q,k)
&=\phi(q)^\top\phi(k),
\qquad \phi(x)=\mathrm{ELU}(x)+1.
\end{aligned}
$$

## 2. Каузальная маска

Определим накопленное состояние ключей и значений и состояние нормировки:

$$
S_t=\sum_{j=1}^{t}v_j\phi(k_j)^\top,
\qquad
z_t=\sum_{j=1}^{t}\phi(k_j).
$$

Перегруппировка сумм даёт

$$
\boxed{
y_t=
\frac{S_t\phi(q_t)}
{\max\left(\phi(q_t)^\top z_t,\varepsilon\right)}
}
$$

Состояния можно обновлять последовательно:

$$
\boxed{
S_t=S_{t-1}+v_t\phi(k_t)^\top,
\qquad
z_t=z_{t-1}+\phi(k_t)
}
$$

при $S_0=0$ и $z_0=0$. Небольшая нижняя граница $\varepsilon$ стабилизирует деление.


## 3. Вычислительная сложность

| Ядро внимания | Вычислительная сложность |
| --- | ---: |
| Softmax attention | $O(N^2d)$ |
| Linear attention | $O(Nd^2)$ |


## 4. Сочетание линейного внимания с RoPE

Применим зависящие от позиции вращения после преобразования признаков:

$$
\tilde q_t=R_t\phi(q_t),
\qquad
\tilde k_j=R_j\phi(k_j),
$$

где $R_t$ — матрица вращения RoPE в позиции $t$. Тогда

$$
y_t^{\mathrm{RoPE}}=
\frac{\sum_{j\le t}(\tilde q_t^\top\tilde k_j)\,v_j}
{\max\left(\phi(q_t)^\top\sum_{j\le t}\phi(k_j),\varepsilon\right)}.
$$

## 5. Вывод

> В наших экспериментах linear attention несколько уступает softmax по качеству на TinyStories. В конфигурации с контекстом 8192 он сокращает среднее время обучения на батч и повышает скорость генерации. На контексте 512 наша реализация linear attention работает медленнее softmax.
