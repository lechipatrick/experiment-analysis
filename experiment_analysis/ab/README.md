## User A/B experiments

### Additive metric

There are 3 inference methods: ztest, randomization, and bootstrap. To describe these, we need a bit of notation.

There are $n$ units. Each unit is assigned to either control group $C$ or treatment group $T$. A metric $x_i$ is observed for each unit $i$.

#### ztest
The estimation framework is that of Neyman's repeated sampling approach (see Causal Inference for Statistic, Social, and Biomedical Sciences by Imbens and Rubins).

Denote the metric mean in group $G$ as $\bar x_G = \frac{1}{|G|} \sum_{i \in G} x_i$, where $G$ can be $C$ or $T$.

Denote the metric variance in group $G$ as $s^2_G = \frac{1}{|G|} \sum_{i \in G} (x_i - \bar x_G)^2$.

Treatment effect is estimated as difference in means $\hat \tau = \bar x_T - \bar x_C$

Estimated variance of the difference in means $\hat v = \frac{1}{|T|} s^2_T + \frac{1}{|T|} s^2_C$

Under certain assumptions, $\hat \tau$ is a good estimator of the true treatment effect (unbiased and consistent), and has variance $\hat v$.

The z statistic is defined to be $z = \frac{\tau}{\sqrt{\hat v}}$. Under the null, the z statistic would be normally distributed with zero mean and unit variance. 

p-value is computed based on how far the z statistic strays from this $N(0,1)$ distribution.

#### Ratio metric