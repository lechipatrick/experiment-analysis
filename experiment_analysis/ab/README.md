# User A/B experiments

## Additive metric

There are 3 inference methods: ztest, randomization, and bootstrap. To describe these, we need a bit of notation.

There are $n$ units. Each unit is assigned to either control group $C$ or treatment group $T$. A metric $x_i$ is observed for each unit $i$.

Treatment effect is defined as the difference between (1) the metric mean in the scenario where every unit is assigned to treatment, and (2) the metric mean where every unit is assigned to control. We are generally interested in whether the treatment effect is non-zero.

Common examples of additive metric: purchase volume, number of shared rides taken, etc.
### ztest
The estimation framework is that of Neyman's repeated sampling approach.

Denote the metric mean in group $G$ as $\bar x_G = \frac{1}{|G|} \sum_{i \in G} x_i$, where $G$ can be $C$ or $T$.

Treatment effect is estimated as difference in means $\hat \tau = \bar x_T - \bar x_C$

Denote the metric variance in group $G$ as $s^2_G = \frac{1}{|G|} \sum_{i \in G} (x_i - \bar x_G)^2$.

Estimated variance of the difference in means $\hat v = \frac{1}{|T|} s^2_T + \frac{1}{|T|} s^2_C$

Under certain assumptions, $\hat \tau$ is a good estimator of the true treatment effect (unbiased and consistent), and has variance $\hat v$.

The z statistic is defined to be $z = \frac{\hat \tau}{\sqrt{\hat v}}$. Under the null, the z statistic would be normally distributed with zero mean and unit variance. 

p-value is computed based on how far the z statistic strays from this $N(0,1)$ distribution.

Confidence interval is based on the point estimate of the treatment effect, the estimated variance.

See Causal Inference for Statistics, Social, and Biomedical Sciences by Imbens and Rubin, chapter 6.

### randomization
Randomization inference assumes a sharp null (treatment effect is exactly zero for every single unit). The procedure involves randomly re-assigning the treatment status (whether a unit is in $C$ or $T$), and computing the treatment effect of the resulting data. Doing so repeatedly yields an empirical distribution of the treatment effect under the null. Comparing the actual, estimated treatment effect against this distribution provides a measure of how unlikely the null is, and is how p-value is calculated.

This approach doesn't lend itself naturally to a notion of confidence interval.

See Causal Inference for Statistics, Social, and Biomedical Sciences by Imbens and Rubin, chapter 5.

### bootstrap
Bootstrap provides an alternate way to computing the variance of the estimator. For additive metric, this is rather trivial, but for other metrics, the variance can be hard to compute. 

The procedure draws, with replacement, from the data until it gets $n$ units. The treatment effect is estimated from this bootstrapped data. Doing so repeatedly yields an empirical distribution of the treatment effect. The variance of this empirical distribution is used to compute the z statistic and the confidence interval.

See An Introduction to the Bootstrap by Efron and Tibshirani.

## Ratio metric
There are 4 inference methods: delta, Fieller, randomization, and bootstrap. To describe these, we need a bit of notation.

There are $n$ units. Each unit is assigned to either control group $C$ or treatment group $T$. A numerator metric $x_i$ and denominator metric $y_i$ are observed for each unit $i$. 

The ratio metric for a group $G$ is defined as $r_G = \frac{\sum_{i \in G} x_i}{\sum_{i \in G} y_i}$.

Treatment effect is defined as the difference between (1) the ratio metric in the scenario where every unit is assigned to treatment, and (2) the ratio metric where every unit is assigned to control. We are generally interested in whether the treatment effect is non-zero.

Common examples of ratio metric: click thru rate (numerator = click, denominator = impression), purchase rate conditional on add to cart (numerator = purchase or not, denominator = add to cart or not), amount of the tip over the total amount of the order, etc.

### randomization
Same as randomization for additive metric.

### bootstrap
Same as bootstrap for additive metric.

### delta method

### Fieller method