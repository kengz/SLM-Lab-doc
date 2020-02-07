# Performance Metrics

The [analysis module](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/experiment/analysis.py) of SLM Lab calculates a set of metrics to measure the performance of an RL agent, which are logged during a run. These metrics can be computed from a time series of returns computed from periodic evaluation.

As motivation, we observe that the returns graph is commonly plotted to present **qualitative** visual comparison among RL algorithms, but it often contains only one **quantitative** result that is the final \(averaged\) return. We acknowledge that a single number is not sufficient to capture all the rich information in a return graph. Qualitatively, the graph visually compares if an algorithm is **stronger**, trains **faster**, is more **stable**, and so on. We turn these into quantitative results by designing metrics that can be compared numerically.

### Prerequisites

Our metrics has two simple prerequisites. The first is the time series from the return graph that is readily available. More precisely, it should record the average return, total frames, and total optimization steps during periodic evaluations, or equivalently

$$
\overline{R}_i, \ frame_i, \ optstep_i, \ \text{for } i = 1, \ldots, N
$$

The average return is computed by averaging over _S_ episodes during evaluation at checkpoint _i_. To balance between the accuracy of the average and computation cost for evaluation, we typically use _S=4,_ and it can be run as parallel sessions in practice.

The second prerequisite is the average return from a random policy, defined as

$$
\overline{R}_{rand} = \mathbb{E}[R_{\pi^{rand}}]
$$

, also known as the random baseline. It is the expectation of the return of a policy with maximum-entropy. Since entropy is maximized for a uniform distribution, the random policy is simply a uniform distribution over the action space. Conveniently, the random policy is already implemented in any environment `env` using the OpenAI gym interface, and can accessed using the method `env.action`_`space.sample()`._ Using this, the random baseline can be obtained easily by running the random policy and averaging the returns over a number of episodes.

## SLM Lab Metrics

Now, we propose four metrics derived from the quantitative notions from a returns graph to measure the performance of an algorithm in an environment. These are **strength**, **efficiency \(speed\)**, **stability**, and **consistency**.

### Strength

The first metric is strength, defined as

$$
\textit{strength: } str = \frac{1}{N} \sum_{i=0}^N \overline{R}_i - \overline{R}_{rand}
$$

This measures the average performance of an algorithm against the random baseline, which is a natural zero-point in its scale. Strength immediately tells if an algorithm is working if _str &gt; 0_, or failing otherwise. Since it is simply the average return with a constant offset, it is familiar and easy to adopt to any evaluation workflow.

Note that the term inside the summation,

$$
\overline{R}_i - \overline{R}_{rand}
$$

is simply the local strength at a specific checkpoint _i_, which measures the strength at a specific frame or optimization step as

$$
str_i = str_{frame_i} = str_{optstep_i}
$$

for example, strength at 10-million frames where _i = 10M_. Given a series of local strengths, the maximum and the minimum,

$$
str_{max}, str_{min}
$$

also capture the best and worst behaviors of an algorithm. Note that strength is simply the mean of local strengths.

The maximum strength captures the peak performance of an algorithm even in the case of regression or policy collapse. The strength, which is proportional to the area under the curve, tells if an algorithm attains its strength quickly and maintains it longer, especially when compared to the maximum strength. Additionally, to obtain a smoother plot, a moving average with a chosen window _w_ \(typically 100\) can be used on top of strength; this is similar to the usual last-100 episode return average used in several papers.

With a proper zero-point, strength is **relative** and **transitive**. If algorithms _A, B_ have strengths 20, 40 respectively, then their strength ratio _B/A_ = 40/20 = 2, hence _B_ is 2 times stronger than _A_. If an algorithm _C_ is 3 times stronger than _B_, we can infer that _C/A =_ _C/B \* B/A_ = 3 \* 2 = 6, and so _C_ is 6 times stronger than _A_. Strength transitivity allows a new algorithm to be compared against all the other algorithms by comparing it against only one standard algorithm of choice. This can make research easier by requiring a researcher to compare against one standard algorithm instead of many.

### Efficiency

The second metric is efficiency, defined with a time unit _t_ as

$$
\textit{efficiency: } e = \frac{\sum_{i=0}^N \frac{1}{t_i} str_i}{\sum_{i=0}^N \frac{1}{t_i}}
$$

Note that it is a weighted average with _1/t_ as the weight. Intuitively, it measures how fast an algorithm attains its strength by assigning higher weight to strength earned early on than later. There are two ways of measuring time _t_ that are of interest to us --- frames and optimization steps. Respectively, they yield:

$$
\textit{sample efficiency: } se = \frac{\sum_{i=0}^N \frac{1}{frame_i} str_i}{\sum_{i=0}^N \frac{1}{frame_i}}
$$

$$
\textit{training efficiency: } te = \frac{\sum_{i=0}^N \frac{1}{optstep_i} str_i}{\sum_{i=0}^N \frac{1}{optstep_i}}
$$

Since frames measures the amount of samples used by an agent to attain its strength, the first equation above measures sample efficiency. Similarly, since optimization steps measures the amount of gradient updates used to attain strength, the second equation captures training efficiency. Like strength, these metrics are also relative and transitive. Efficiency can also be defined point-wise by computing it at every checkpoint _k_ simply by summing from _i = 0, ..., k_ to yield

$$
se_k, te_k
$$

. Moreover, the value range of a weighted average is independent from the choice of checkpoint resolution _frame\_{i+1} - frame\_i_, although coarser resolution will cause some loss in accuracy. Typically, we use a resolution of 50,000 frames.

### Stability

The third metric is stability, defined as

$$
\textit{stability: } sta = 1 - \left| \frac{\sum_{i=0}^{N-1} \min(str_{i+1} - str_i, 0)}{\sum_{i=0}^{N-1} str_i} \right|
$$

This equation measures the average ratio of strength lost between checkpoints as an instability measure, then takes the complement of the ratio so that _sta = 1_ for a perfectly stable algorithm and decreases for more unstable algorithms. It also has the relative and transitive properties. Note that _sta &lt; 0_ can happen when the strength drops too much, for instance during a performance collapse. Note that the definition also works for both positive and negative strengths since it is sign-independent. Stability can also be defined locally for each checkpoint _i = 0, ..., N-1_ as

$$
sta_i = 1 - \left| \frac{\min(str_{i+1} - str_i, 0)}{str_i} \right|
$$

### Consistency

The fourth and the final metric is consistency, defined for multiple series of strengths _str\_{i,j}_ obtained from multiple repeated sessions _j_

$$
\textit{consistency: } con = 1 - \frac{\sum_{i=0}^N 2 stdev_j(str_{i,j})}{\sum_{i=0}^N avg_j(str_{i,j})}
$$

This equation measures the average width of the error band to the average strength, then take its complement so that _con = 1_ for perfectly consistent sessions with 0 error band and decreases for trials with larger error bands. It also has the relative and transitive properties. The error band is defined with simple standard deviation, with the upper and lower bounds as

$$
avg_j(str_{i,j}) + stdev_j(str_{i,j}) \\
avg_j(str_{i,j}) - stdev_j(str_{i,j})
$$

respectively, and the error band as

$$
(avg_j(str_{i,j}) + stdev_j(str_{i,j})) - (avg_j(str_{i,j}) - stdev_j(str_{i,j})) = 2 stdev_j(str_{i,j})
$$

Consistency can also be defined locally for each checkpoint _i = 0, ..., N_ as

$$
con_i = 1 - \frac{2 stdev_j(str_{i,j})}{avg_j(str_{i,j})}
$$

