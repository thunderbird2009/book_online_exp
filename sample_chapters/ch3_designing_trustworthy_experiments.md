# Chapter 3: Designing Trustworthy Experiments

Chapter 2 built the statistical engine of experimentation: hypothesis testing, p-values, power, and sample size. We now have the tools to determine statistical significance. However, a statistically sound result is unreliable if the experiment is poorly designed. Flawed designs lead to biased outcomes, incorrect conclusions, and bad product decisions.

This chapter transitions from the "what" of statistical analysis to the "how" of practical experiment design. We cover common A/B test architectural patterns, walk through a step-by-step design process, and identify critical pitfalls that can sabotage findings. By the end, you will architect experiments that generate trustworthy, actionable insights [1][2].

## Table of Contents

- [1. Common A/B Test Structures](#1-common-ab-test-structures)
    - [1.1. Two-Sample Test (Independent Samples)](#11-two-sample-test-independent-samples)
        - [One-Sided vs. Two-Sided Test Calculations](#one-sided-vs-two-sided-test-calculations)
    - [1.2. The Paired Test (Before-and-After Design)](#12-the-paired-test-before-and-after-design)
        - [P-value and Test Statistic using T-tests](#p-value-and-test-statistic-using-t-tests)
        - [Sample Size Calculation](#sample-size-calculation)
    - [1.3. Non-Inferiority Tests (Proving "No Harm")](#13-non-inferiority-tests-proving-no-harm)
        - [P-value and Test Statistic using T-Test](#p-value-and-test-statistic-using-t-test)
        - [Sample Size Calculation](#sample-size-calculation-1)
- [2. Putting It All Together: A Step-by-Step Experiment Design](#2-putting-it-all-together-a-step-by-step-experiment-design)
    - [2.1. Step 1: The Business Question and Metric Definition](#21-step-1-the-business-question-and-metric-definition)
    - [2.2. Step 2: Formulating the Hypotheses](#22-step-2-formulating-the-hypotheses)
    - [2.3. Step 3: Designing the Experiment (Parameter Selection)](#23-step-3-designing-the-experiment-parameter-selection)
    - [2.4. Step 4: Calculating the Required Sample Size](#24-step-4-calculating-the-required-sample-size)
    - [2.5. Step 5: Defining the Fixed Horizon and Decision Rule](#25-step-5-defining-the-fixed-horizon-and-decision-rule)
- [3. Common Pitfalls and How to Avoid Them](#3-common-pitfalls-and-how-to-avoid-them)
    - [3.1. The Novelty Effect and Learning Effects](#31-the-novelty-effect-and-learning-effects)
        - [How to Mitigate Novelty and Learning Effects](#how-to-mitigate-novelty-and-learning-effects)
    - [3.2. The Multiple Testing Problem](#32-the-multiple-testing-problem)
        - [The Problem of Multiple Metrics](#the-problem-of-multiple-metrics)
        - [The Problem of Repeated Experiments](#the-problem-of-repeated-experiments)
    - [3.3. Peeking at Results](#33-peeking-at-results)
- [4. Summary and Transition](#4-summary-and-transition)
- [5. References and Further Reading](#5-references-and-further-reading)

---

## 1. Common A/B Test Structures

While the foundational theory of hypothesis testing is general, the specific structure of your hypotheses and the way you collect data depend on the question you are trying to answer. This section covers the most common experimental designs. It is crucial to distinguish these *designs* from the *statistical tests* discussed in the previous section. The experimental design is about the "how" of the setup (e.g., independent groups vs. paired samples), while the metric type determines the specific statistical engine (e.g., t-test vs. z-test) used for the analysis.

### 1.1. Two-Sample Test (Independent Samples)
This is the classic experimental design and the workhorse of product experimentation. It is used to compare the means or proportions of two independent groups to determine if they are statistically different from each other.

-   **Structure:** Users are randomly assigned to one of two groups: Control (A) or Treatment (B). The groups are **independent**, meaning one user's assignment has no bearing on another's, and a user is only in one group.
-   **Hypotheses (Two-Sided):**
    -   $H_0: \mu_A = \mu_B$ (The metric is the same in both groups)
    -   $H_a: \mu_A \neq \mu_B$ (The metric is different between the groups)
-   **Analysis:** The analysis uses the appropriate two-sample test for the metric type.
    -   For a **continuous metric** (like revenue), a **two-sample Welch's t-test** is used.
    -   For a **binomial metric** (like conversion rate), a **two-sample z-test for proportions** is used.
-   **Use Case:** This is the default choice for answering most product questions. Is a new headline better than the old one? Does a new recommendation algorithm drive more revenue?

#### One-Sided vs. Two-Sided Test Calculations
Here we assume T-test for the metric. The core formulas for the test statistic and sample size were detailed in the previous sections. However, they differ slightly depending on whether you are running a one-sided or two-sided test, which involves a critical trade-off between statistical power and the ability to detect unexpected outcomes.

-   **P-value:** For a given z-score, the p-value for a one-sided test is half that of a two-sided test. This is because you are only measuring the area in one tail of the distribution.

-   **Sample Size:** The sample size formula uses a different critical value for alpha.
    -   **Two-sided:** Uses $z_{\alpha/2}$ (e.g., 1.96 for $\alpha=0.05$).
    -   **One-sided:** Uses $z_{\alpha}$ (e.g., 1.645 for $\alpha=0.05$).
    
**The Trade-Off: Power vs. Safety**

-   **Pro of One-Sided Tests:** Because $z_{\alpha}$ is smaller than $z_{\alpha/2}$, a one-sided test requires a smaller sample size to achieve the same power. This makes it "easier" to find a statistically significant result in the direction you are testing for.

-   **Con of One-Sided Tests:** This statistical power comes at a high price: **a one-sided test is blind to effects in the opposite direction.** If you are testing for a positive lift ($H_a: \mu_B > \mu_A$) but your feature actually causes a significant *decrease* in the metric, the test will simply produce a large p-value and you will "fail to reject the null." You will not be able to distinguish between "no effect" and "a significant harmful effect." This is extremely dangerous in product development, where understanding downside risk is often just as important as measuring upside potential.

For this reason, the **two-sided test is the standard and strongly recommended choice for the vast majority of A/B tests.** It provides a crucial safety net, ensuring that you are alerted to statistically significant outcomes in either direction, whether they are positive or negative.

### 1.2. The Paired Test (Before-and-After Design)
A paired test is used when the observations in the two groups are not independent but are instead naturally paired [3]. The most common scenario for this is a "before-and-after" study on the same set of users.

-   **Structure:** Instead of having a separate control and treatment group, you have a single group of users. You measure a metric for them *before* a change is introduced, and then you measure the same metric for the *same users* after the change. The "pairs" are the two measurements from the same user.
-   **How it Works:** The test does not analyze the two groups of measurements independently. Instead, it first calculates the **difference** for each pair (e.g., `user1_after - user1_before`, `user2_after - user2_before`, etc.). It then performs a **one-sample t-test** on this new list of differences, testing if the mean of these differences is statistically different from zero.
-   **Hypotheses:**
    -   $H_0: \mu_{difference} = 0$ (The mean of the before-and-after differences is zero)
    -   $H_a: \mu_{difference} \neq 0$ (The mean of the differences is not zero)
-   **Advantage: Variance Reduction:** This is a powerful technique for reducing variance [4]. By using each user as their own control, you eliminate inter-user variability and isolate the effect of the treatment. This often results in a much more powerful test that requires a smaller sample size.

#### P-value and Test Statistic using T-tests
A one-sample t-test is performed on the list of differences. The test statistic is calculated as:
$$
t = \frac{\bar{d} - 0}{s_d / \sqrt{n}}
$$
Here, $\bar{d}$ is the mean of the differences, $s_d$ is the standard deviation of the differences, and $n$ is the number of pairs. The p-value is derived from this t-statistic using the t-distribution.

#### Sample Size Calculation
The calculation is performed for a one-sample test on the list of differences.
$$
n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot \sigma_d^2}{(\text{MDE})^2}
$$
The key difference is $\sigma_d^2$, the variance of the *differences*. This can be dramatically smaller than the variance of the original metric, making the required sample size ($n$) much lower.

### 1.3. Non-Inferiority Tests (Proving "No Harm")
Non-inferiority testing flips the standard hypothesis structure [3]. It is used when your goal is not to prove that a new version is *better*, but to prove that it is *not unacceptably worse*.

-   **Structure:** The experiment setup is the same as a standard A/B test with two independent groups. The key difference is in the hypothesis formulation.
-   **The "Non-Inferiority Margin" ($\delta$):** Before the test, you must define a margin, $\delta$, which is the largest decrease in performance you are willing to tolerate. This is a business decision. For example, you might be willing to accept a 0.1% drop in conversion rate for a large infrastructure improvement.
-   **Hypotheses:**
    -   $H_0: \mu_A - \mu_B \ge \delta$ (The new version B is unacceptably worse than A)
    -   $H_a: \mu_A - \mu_B < \delta$ (The new version B is **not** unacceptably worse than A)
-   **Analysis:** The analysis uses a one-sided t-test or z-test, but the null hypothesis is centered on the margin $\delta$ instead of 0.
-   **Use Case:** Essential for engineering-driven changes like infrastructure migrations, code refactoring, or third-party vendor swaps. The goal is to ship technical improvements while guaranteeing they do not significantly harm user-facing metrics.

#### P-value and Test Statistic using T-Test
The test statistic is shifted by the non-inferiority margin, $\delta$. For a metric where a higher value is better, the z-score is:
$$
z = \frac{(\bar{x}_B - \bar{x}_A) - (-\delta)}{\text{Standard Error}}
$$
The p-value is then calculated from one tail of the standard normal distribution based on this z-score. It represents the probability of observing the data if the new version were truly inferior by at least $\delta$.

#### Sample Size Calculation
The formula is adjusted for the one-sided nature of the hypothesis.
$$
n = \frac{(z_{\alpha} + z_{\beta})^2 \cdot (\sigma_A^2 + \sigma_B^2)}{(\text{MDE} - \delta)^2}
$$
-   We use $z_{\alpha}$ instead of $z_{\alpha/2}$ because it's a one-sided test (e.g., for $\alpha=0.05$, $z_{\alpha}$ is 1.645).
-   The MDE is often set to 0, as the goal is simply to show the new version is not worse by more than the margin $\delta$.

---

## 2. Putting It All Together: A Step-by-Step Experiment Design

This chapter has covered the individual components of statistical testing. Now we synthesize them by walking through a complete, practical example of designing an experiment from start to finish. This process serves as a blueprint for how to apply the theory we've learned.

**Scenario:** We are a product team at an e-commerce company. Our goal is to improve the number of users who complete a purchase.

### 2.1. Step 1: The Business Question and Metric Definition

-   **Business Question:** We hypothesize that changing the color of our main "Add to Cart" button from its current grey to a more vibrant orange will draw more user attention and lead to more users adding items to their cart.
-   **Primary Metric:** We define our key success metric as the **Add to Cart Conversion Rate**. This is a proportion metric, calculated as: `(Number of unique users who click "Add to Cart") / (Number of unique users who view a product page)`.

### 2.2. Step 2: Formulating the Hypotheses

We need to state our question in the formal language of hypothesis testing. Since we want to detect if the change has *any* significant impact (positive or negative), we choose a two-sided test.

-   Let $p_A$ be the true conversion rate for the control group (grey button).
-   Let $p_B$ be the true conversion rate for the treatment group (orange button).

-   **Null Hypothesis ($H_0$):** The button color has no effect on the conversion rate.
    $$ H_0: p_B = p_A $$
-   **Alternative Hypothesis ($H_a$):** The button color has an effect on the conversion rate.
    $$ H_a: p_B \neq p_A $$

### 2.3. Step 3: Designing the Experiment (Parameter Selection)

This is the most critical design phase, where we define our risk tolerance and what we consider a meaningful result.

-   **Significance Level ($\alpha$):** We will use the industry standard, setting our tolerance for a false positive to 5%.
    -   $\alpha = 0.05$, which for a two-sided test gives us a critical value $z_{\alpha/2} = 1.96$.
-   **Statistical Power ($1-\beta$):** We want a high probability of detecting a real effect if one exists. We choose the industry standard of 80% power.
    -   $1-\beta = 0.80$ (so $\beta = 0.20$), which gives us a critical value $z_{\beta} = 0.84$.
-   **Baseline Rate:** We analyze historical data and find that our current average Add to Cart conversion rate is 4.0%.
    -   $p_A = 0.04$.
-   **Minimum Detectable Effect (MDE):** This is a product and business decision. After discussion, the team decides that a 10% relative lift would be meaningful enough to justify the change.
    -   Absolute MDE = $0.04 \times 10\% = 0.004$.
    -   This means we want our experiment to be sensitive enough to detect a change if the new conversion rate is $p_B = 4.4\%$ or $3.6\%$.

### 2.4. Step 4: Calculating the Required Sample Size

Now we assemble our parameters into the sample size formula for a two-sample test of proportions.

$$
n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot (p_A(1-p_A) + p_B(1-p_B))}{(p_A - p_B)^2}
$$

Plugging in our values:
-   $z_{\alpha/2} = 1.96$
-   $z_{\beta} = 0.84$
-   $p_A = 0.04$
-   $p_B = 0.044$ (our target for detection)
-   $p_A - p_B = -0.004$

$$
n = \frac{(1.96 + 0.84)^2 \cdot (0.04(1-0.04) + 0.044(1-0.044))}{(-0.004)^2}
$$
$$
n = \frac{(2.8)^2 \cdot (0.0384 + 0.042064)}{0.000016} = \frac{7.84 \cdot 0.080464}{0.000016} \approx 39,427
$$

The result is that we need approximately **39,427 users per group**. The total experiment will need about 78,854 users.

### 2.5. Step 5: Defining the Fixed Horizon and Decision Rule

This final step is crucial for maintaining the statistical integrity of the experiment.

-   **Fixed Horizon:** Based on our calculation, we commit to running the experiment until we have collected data from at least **39,427 users in the control group and 39,427 users in the treatment group**. We will not stop the test early or make a decision based on partial results. This commitment is our primary defense against the "peeking" problem and regression to the mean. If our site gets 20,000 eligible users per day, we can estimate the experiment will need to run for about 4 days.

-   **Decision Rule:** We pre-specify our conditions for success. As established in our discussion on the duality of p-values and confidence intervals, for a two-sided test, a p-value less than 0.05 is mathematically equivalent to a 95% confidence interval for the *difference between the two conversion rates* that does not contain zero. One outcome implies the other. Our decision rule is therefore:
    1.  After the fixed horizon is reached, calculate the p-value and the 95% confidence interval for the difference in conversion rates.
    2.  If **p < 0.05**, the result is statistically significant. The 95% confidence interval will not contain zero (e.g., `[+0.1%, +0.7%]`). We conclude the new button has a significant effect, and we will launch the orange button.
    3.  If **p $\ge$ 0.05**, the result is not statistically significant. The 95% confidence interval will contain zero (e.g., `[-0.2%, +0.6%]`). We conclude there is not enough evidence that the new button has an effect, and we will stick with the original grey button.

This complete design provides a clear plan of action and a rigorous, data-driven framework for making a decision, tying together all the statistical concepts discussed in this chapter into a single, actionable process.

---

## 3. Common Pitfalls and How to Avoid Them

Statistical intuition is not just about knowing the formulas; it's about recognizing the patterns and scenarios where those formulas can mislead us. This section covers the most common traps in experimental analysis.

### 3.1. The Novelty Effect and Learning Effects

Human behavior is not static. Users react to change, and this reaction can create temporary distortions in experiment results. These distortions typically manifest in two ways: the novelty effect and the learning effect.

| **Aspect** | **Novelty Effect** | **Learning Effect** |
|------------|-------------------|---------------------|
| **Definition** | Temporary engagement increase because change is new | Temporary performance decrease due to disrupted workflow |
| **Direction** | Initial positive lift → decline to baseline | Initial negative dip → recovery to higher level |
| **Example** | Button color change: CTR spikes then returns to baseline | Navigation redesign: task time increases then improves beyond old baseline |
| **Who Affected** | Existing users (new users immune) | Existing users (new users immune) |
| **Risk if Stopped Early** | Launch based on temporary lift; long-term metrics show no real improvement | Kill genuinely better feature based on temporary learning curve |

#### How to Mitigate Novelty and Learning Effects
The primary strategy is to **run experiments long enough for behavior to stabilize.**

1.  **Cohort Analysis:** Segment results by user tenure and time. Analyze new users separately (immune to both effects) for a cleaner signal. Plot day-over-day trends: novelty effects show high initial lift decaying to stable baseline; learning effects show initial dip recovering to higher stable level.

2.  **Extended Duration:** Run experiments for multiple weeks on core UI changes. Standard one-two week tests are often insufficient.

3.  **Reverse Experiments:** Post-launch, revert a treatment group to the old experience. A significant metric drop confirms the improvement was real, not temporary.

Experimentation platforms should provide cohort analysis tools, allowing easy data slicing by user acquisition date to diagnose these effects.

### 3.2. The Multiple Testing Problem

The multiple testing problem is one of the most insidious and common ways that experimenters draw false conclusions from data. It stems from a simple fact: the significance level, $\alpha=0.05$, guarantees a 5% false positive rate **for a single statistical test**. When you run multiple tests, the probability of getting at least one false positive across the entire set of tests increases dramatically.

This problem manifests in several ways.

#### The Problem of Multiple Metrics
The first form of the multiple testing problem occurs when an experiment tracks many different metrics, and the experimenter is willing to declare victory if *any* of them shows a statistically significant lift. If an experiment tracks 20 different metrics, it's highly probable that at least one of them will show a p-value less than 0.05 just by random chance.

**The Solution: Metric Hierarchy**

Establish a clear hierarchy *before* experiment launch:

| **Metric Type** | **Purpose** | **Decision Rule** | **Example** |
|----------------|-------------|-------------------|-------------|
| **Primary (1-2 max)** | Defines success | Launch decision based **exclusively** on these | Add to Cart Conversion Rate |
| **Secondary** | Observational context | Treat significant results as new hypotheses for future experiments | Revenue Per User, Purchase Rate |
| **Guardrail** | Prevent unintended harm | Significant negative change blocks launch, even if primary improves | Page Load Time, Error Rate, Churn |

This approach is the main defense against multiple testing. While statistical corrections like Bonferroni exist, they are too conservative for product experimentation [1][5].

#### The Problem of Repeated Experiments
A second form of this problem is re-running an entire experiment hoping for a different result.

*   **Scenario:** A team runs an experiment for a new feature, and it fails to show a statistically significant result. Convinced the feature is a good idea, they run the exact same experiment again a month later. It fails again. They try a third time, and it finally shows a p-value of 0.04.
*   **The Flaw:** This is a clear case of multiple testing. Each time the experiment is re-run, it's another independent statistical test. If you are willing to run the test multiple times, you are giving yourself multiple opportunities to hit a 5% false positive.
*   **The Solution:** A non-significant result should be taken as evidence that there is no detectable effect *with the current design*. Instead of re-running the same experiment, the team should iterate on the feature to create a *stronger* treatment (e.g., a more noticeable UI change) that would have a larger effect size and a better chance of being detected in a *new* experiment.

### 3.3. Peeking at Results

This is the act of repeatedly checking an experiment's results over time with the intention of stopping as soon as statistical significance is reached [6].

The significance level, $\alpha = 0.05$, is guaranteed *only if you commit to checking the results exactly once* after the pre-calculated sample size has been reached. Each "peek" before that point is an invalid, premature statistical test.

**The Peeking Workflow:**
1.  Launch experiment
2.  Check results dashboard hourly/daily
3.  Ask: "Is p < 0.05 yet?"
4.  If yes → stop and declare winner
5.  If no → continue and check again

**Why It's Tempting (and Flawed):**
- **Impatience:** Teams want quick wins instead of waiting for full experiment duration
- **P-value Misinterpretation:** Treating p-value as a real-time "truth meter" when it's only valid at pre-specified sample size
- **Ignoring Fixed Horizon:** Viewing sample size as an estimate rather than a strict prerequisite

The p-value fluctuates randomly over time. Checking 20 times gives random chance 20 opportunities to dip below 0.05 purely by chance. With 20 peeks, the false positive rate inflates from 5% to over 30%!

**The Consequence:**
Peeking leads to launching features that have no real effect. An experimenter checks the results daily. On day 3, the p-value is 0.25. On day 4, it's 0.12. On day 5, due to random fluctuation, it dips to 0.04. The experimenter, excited, stops the test, declares victory, and launches the feature. In reality, they just got "lucky" on one of their peeks. Had they let the experiment run its full course, the p-value would likely have regressed back to a non-significant level.

**How to Avoid It:**
As engineers building and using experimentation platforms, we must implement safeguards to protect against the dangers of multiple testing.

1.  **Fixed Horizon (Pre-computation):** The simplest and most robust method is to agree on the sample size *before* the experiment starts and to only analyze the results once that sample size has been reached. This enforces the "single look" model for which the statistical tests were designed. Your platform's sample size calculator is the key tool here.
2.  **Sequential Testing (Advanced):** For teams that cannot afford to wait for a fixed horizon, more advanced statistical methods exist that are designed to handle multiple looks. These methods, such as the **Sequential Probability Ratio Test (SPRT)** or other alpha-spending functions, adjust the significance boundary at each peek. For example, you might need a p-value of < 0.01 to stop on day 1, < 0.02 on day 3, and so on, until you reach the final boundary of 0.05 at the planned end of the experiment. These methods are more complex to implement correctly but can allow for faster decisions without inflating the Type I error rate. A full exploration of these techniques is reserved for Part III (Chapter 11).
3.  **Hold-out / Hold-back Sets (Validation):** Another robust method is to use a hold-out set. When an experimenter stops an experiment early based on a peek, the platform can automatically continue running the experiment on a small, separate "hold-out" group of users. The results from the initial experiment are then validated against this hold-out group. If the effect is real, it should persist in the hold-out. If it was a false positive from peeking, it will likely disappear.

For most platforms, enforcing a **fixed horizon** is the most practical and effective solution. Building a culture where experimenters trust their sample size calculations and resist the urge to peek is one of the most important steps toward running a trustworthy experimentation program. The platform's UI can help by de-emphasizing or even hiding the p-value until the experiment has reached its required sample size.

---

## 4. Summary and Transition

In this chapter, we moved from the abstract statistical theory of Chapter 2 to the concrete practice of designing and running experiments. We have seen that building a trustworthy experiment requires more than just correct formulas; it demands a disciplined process and a keen awareness of potential pitfalls.

We covered three key areas:
1.  **Common Experimental Designs:** We explored the workhorse of A/B testing—the independent two-sample test—and contrasted it with powerful alternatives like paired tests for variance reduction and non-inferiority tests for safe engineering rollouts.
2.  **A Step-by-Step Design Process:** We walked through a practical, end-to-end blueprint for designing an experiment, from defining the business question and primary metric to calculating sample size and pre-specifying the decision rule.
3.  **Common Pitfalls:** We learned to identify and mitigate the subtle but dangerous traps that can invalidate results, including the novelty and learning effects, the multiple testing problem, and the temptation to peek at results.

The central theme of this chapter is that **rigor precedes results**. A disciplined, structured approach to experiment design is the only way to ensure that the data we collect is a reliable foundation for decision-making.

Having established the principles of both statistical analysis and experimental design, we have built a solid foundation for running trustworthy tests. However, the speed and sensitivity of our experiments are often limited by a powerful, invisible force: variance. High variance in our metrics can obscure real effects, forcing us to run experiments for weeks or even months to get a clear signal.

In the next chapter, "Metric Design and Variance Reduction," we will tackle this challenge head-on. We will explore how the very design of a metric can dramatically impact its variance and, consequently, the speed of experimentation. We will introduce powerful techniques to engineer metrics that are both sensitive to change and robust to noise, allowing us to run faster, more conclusive experiments and accelerate the pace of innovation.

---

## 5. References and Further Reading

[1] Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press. – Comprehensive coverage of experimental design, common pitfalls, and best practices for A/B testing at scale.

[2] Fisher, R. A. (1935). *The Design of Experiments*. Oliver and Boyd. – The foundational work on experimental design, introducing principles of randomization, replication, and blocking that underpin modern A/B testing.

[3] Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters: Design, Innovation, and Discovery* (2nd ed.). Wiley. – Classic textbook covering paired designs, non-inferiority testing, and practical experimental design strategies.

[4] Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data." *WSDM '13: Proceedings of the Sixth ACM International Conference on Web Search and Data Mining*. – Discusses variance reduction techniques and the importance of metric design for experiment sensitivity.

[5] Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society, Series B*, 57(1), 289-300. – Foundational paper on addressing the multiple testing problem with false discovery rate control.

[6] Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2017). "Peeking at A/B Tests: Why it Matters, and What to Do About It." *KDD '17: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. – Addresses the statistical dangers of early stopping and continuous monitoring in online experiments.
