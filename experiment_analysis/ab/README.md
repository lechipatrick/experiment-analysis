## User A/B experiments

### Additive metric

There are 3 inference methods: ztest, randomization, and bootstrap. To describe these, we need a bit of notation.

There are $n$ units. Each unit is assigned to either control group $C$ or treatment group $T$. A metric $x_i$ is observed for each unit $i$.

#### ztest
treatment effect $\tau = \frac{1}{|T|} \sum_{i \in T} x_i - \frac{1}{|C|} \sum_{i \in C} x_i$


#### Ratio metric