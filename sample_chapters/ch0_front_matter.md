# Engineering Online Experimentation: Architecture, Pipelines, and Statistical Methods for Production-Scale Systems

In today's data-driven world, **online experimentation** is the engine of product innovation. A/B testing—the most widely used form of online controlled experiments—has become essential for making data-driven decisions about product changes, feature launches, and ML model deployments. While many books explain the statistical theory behind experimentation, a critical gap remains: the practical, hands-on guide for the engineers who build and maintain the systems that make it all possible.

This book is written for you: the **Data Engineer** tasked with building scalable experimentation platforms and pipelines, and the **Machine Learning Engineer** responsible for validating models in production.

## Standing on the Shoulders of Giants

We stand on the shoulders of giants. The field of online experimentation has been shaped by foundational practitioner guides like *Trustworthy Online Controlled Experiments* by Kohavi, Tang, and Xu, and design-oriented works like *Designing with Data* by King, Churchill, and Tan. The statistical theory itself rests on a century of progress, detailed in classic academic texts such as *Statistical Inference* by Casella and Berger and *Probability and Statistics* by DeGroot and Schervish.

This book does not aim to replace those essential resources. Instead, it is designed to answer the question that naturally follows for any engineer studying them: "This is great, but how do I actually *build* it?"

Where other books masterfully cover the 'what' and the 'why'—from foundational statistical theory to product strategy—we dive deep into the 'how.' Our focus is on the code, the architectural patterns, and the implementation details tailored for engineers on the ground. Think of this as the practical, technical companion that bridges the gap between statistical theory and production systems.

## Who This Book Is For

- Data Engineers building reliable, scalable experimentation data flows (assignment, logging, warehousing, stats, reporting)
- ML Engineers training, validating and evaluating models in production, with online experimentation data.
- Platform/Software Architects designing experimentation services and governance
- Product/Data Scientists who want a deeper understanding of systems-level constraints and implementation details of online experimentation.

If you are responsible for moving online experimentation from slides to systems, this book is written for you.

## How to Read This Book

- If you are new to experimentation, read Part I (Ch1–Ch4) in order.
- If you are building a platform, focus on Part II (Ch5–Ch10), then keep Part III as a design reference.
- If you work on ML systems, read Part I, skim Part II, then dive into Part IV (Ch16–Ch19).
- You can also jump directly to topics from the Table of Contents—each chapter stands alone. See also “How to Use This Book” below for role-specific paths.

## What’s Not Covered

- Full statistical proofs and measure-theoretic foundations (we cite and link to canonical sources instead)
- Non-Python code stacks (examples use Python, SQL, and widely used data tools)
- Vendor-specific configuration for commercial platforms (content is vendor-agnostic; principles transfer)
- Deep causal inference beyond what’s needed for product experimentation (e.g., full treatment of IV/DiD/Synthetic Control is out of scope; we provide pointers)
- Reinforcement learning beyond core multi-armed bandits and Bayesian optimization patterns

## Code and Resources

- Companion notebooks and code live in this repository under [code/](code/)
- Chapter images/figures: [Images/](Images/)
- Checklists, templates, and publishing notes: [instru/](instru/)
- External repository (when public): replace the placeholder link in “Prerequisites → Companion Code Repository” with the final URL
- Recommended libraries: SciPy, statsmodels, pandas, numpy; pipelines with dbt/Airflow/Kafka (examples are platform-agnostic)

Tip: Pin your environment to ensure examples run reproducibly (e.g., a requirements.txt or conda environment file alongside notebooks).

## Versioning Policy

- Editions follow semantic versioning: MAJOR.MINOR.PATCH
	- MAJOR (1.0.0, 2.0.0): structural changes, new parts/chapters, or backwards-incompatible reframing
	- MINOR (1.1.0): new sections/examples, substantial clarifications, additional figures
	- PATCH (1.0.1): copyedits, typo/figure fixes, minor clarifications
- The authoritative change log is maintained in [instru/publish.md](instru/publish.md)
- Code samples carry inline version notes when behavior changes across editions
- Citations: include the book version (e.g., “v1.0.0”) in docs and internal references to avoid ambiguity

## What You'll Learn

While the title emphasizes A/B testing as the most familiar entry point, this book covers the full spectrum of online experimentation methods essential for modern data and ML engineering:

*   **Core A/B Testing**: The statistical foundations, platform architecture, and pipeline engineering for standard two-variant experiments
*   **Advanced Experimental Designs**: Multi-variant testing, interleaving for ranking systems, switchback and geo experiments, and sequential testing
*   **Adaptive Methods**: Multi-armed bandits and Bayesian optimization for efficient hyperparameter tuning
*   **ML-Specific Techniques**: Evaluating ML models, counterfactual learning from experiment data, and safe model deployment

Our focus is not on abstract theory, but on the concrete "how-to" of implementation. Inside, you will find:

*   **Architectural patterns** for building robust experimentation platforms with tools like Kafka, dbt, and Airflow
*   **Practical strategies** for testing ML models, from feature flags to managing online/offline evaluation
*   **Real-world case studies** from tech giants like Google, Netflix, Microsoft, and LinkedIn that highlight the engineering challenges and solutions behind their experimentation cultures
*   **Production-ready code examples** for implementing assignment services, ETL pipelines, statistical engines, and data quality checks

This book is designed to be a technical companion, a guide that lives on your digital bookshelf, ready to be consulted when you need to move from concept to code. It is structured to take you from the statistical foundations all the way to deploying advanced ML testing strategies.

---

# Table of Contents

## [Part I: The Statistical and Foundational Core](#part-i-the-statistical-and-foundational-core)

This section establishes the language, theory, and requirements for a robust experimentation system. It ensures engineers understand the "why" and "how" of making valid inferences across all experimental designs.

### [Chapter 1: The Experimentation Mindset](ch1_experimentation_mindset.md)
* From Intuition to Data: The value proposition of online controlled experiments
* Experimentation Terminology: OEC, Guardrail Metrics, Unit of Diversion
* The Experimentation Hierarchy: A/B Tests, A/B/n, MVT, Interleaving, Bandits

### [Chapter 2: The Statistical Engine of Experimentation](ch2_statistical_theory.md)
* Hypothesis Testing: H₀, Hₐ, One-sided vs. Two-sided tests
* Errors and Power: Alpha (α), Beta (β), Type I/II Errors, Statistical Power (1-β)
* P-values and Confidence Intervals: The duality and practical interpretation
* Sample Size Estimation: The four inputs (Baseline, MDE, α, Power)

### [Chapter 3: Designing Trustworthy Experiments](ch3_designing_trustworthy_experiments.md)
* Common Experimental Designs: Two-Sample, Paired, Non-Inferiority
* A Step-by-Step Guide to Experiment Design
* Common Pitfalls: Novelty/Learning Effects, Multiple Testing, Peeking

### [Chapter 4: Metric Design and Variance Reduction](ch4_variance_reduction.md)
* Selecting the OEC: Leading vs. Lagging indicators
* Guardrail Metrics: Protecting the user experience and the business
* The Variance Problem: Why high variance makes tests slow
* Variance Reduction Techniques: CUPED (Controlled-Experiment Using Pre-Experiment Data)

---

## [Part II: Platform Engineering: Building a Production Experimentation System](#part-ii-platform-engineering)

This section details the infrastructure, pipelines, and data quality requirements—the core responsibilities of the Data Engineer. You'll learn how to build a scalable experimentation platform that can handle thousands of concurrent experiments.

### [Chapter 5: Architecture of an Experimentation Platform](ch5_architecture_of_an_experimentation_platform.md)
* The End-to-End View: A high-level system diagram
* Core Components: Assignment Service, Event Ingestion, Data Warehouse, Stats Engine, Management UI
* The Flow of Data: From user assignment to the final results dashboard
* **Industry Spotlight:** eBay's Experimentation Platform (ExP)

### [Chapter 6: User Identity, Diversion, and Segmentation](ch6_user_identity_diversion_segmentation.md)
* The Unit of Diversion: Choosing the right entity (User, Session, Device)
* Randomization and Hashing: Ensuring truly random, stable assignment logic
* Layering and Mutual Exclusion: Running multiple, non-conflicting experiments in parallel
* Handling Identity Stitching and Cross-Device Consistency
* Exclusion Criteria: Dealing with internal users, bots, and bad traffic

### [Chapter 7: Instrumentation and Event Design](ch7_instrumentation_and_event_design.md)
* The Assignment Event: Logging user assignment—when, where, and how
* Designing Context-Rich Events: Ensuring metrics can be sliced by experiment context
* Platform Requirements: Logging libraries, event schemas, and validation

### [Chapter 8: The ETL/ELT Pipeline and Statistical Engine](ch8_pipeline_and_stats_engine.md)
* ETL/ELT for Experiment Data: Joining assignment logs with performance events
* Data Aggregation: Calculating user-level and group-level metrics
* Automating Statistical Analysis: Integrating Python statsmodels/SciPy
* Implementation with dbt: Modeling experiment data

### [Chapter 9: Data Quality and Health Checks](ch9_data_quality_and_health_checks.md)
* Sample Ratio Mismatch (SRM): Detecting biased traffic allocation
* Pre-Experiment Health Checks: Catching issues before they scale
* Intent-to-Treat (ITT) vs. Per-Protocol Analysis
* **Industry Spotlight:** Microsoft's Experimentation Platform and SRM Detection

### [Chapter 10: Deployment and Release Strategies](ch10_deployment_and_release_strategies.md)
* Feature Flagging for Experimentation: Technical implementation
* Canary Deployments vs. A/B Tests: Stability vs. business impact
* **Industry Spotlight:** Netflix's culture of experimentation and progressive rollouts

---

## [Part III: Beyond Basic A/B Testing: Advanced Experimental Designs](#part-iii-advanced-experimental-designs)

This section covers sophisticated online experimentation techniques for complex scenarios. You'll learn specialized designs like interleaving for ranking systems, switchback experiments for spillover effects, multi-armed bandits for dynamic optimization, and methods to accelerate experiments.

### [Chapter 11: Accelerating Experiments and Analyzing Complex Metrics](ch11_accelerating_experiments_and_analyzing_complex_metrics.md)
* Sequential Testing: The statistically valid way to monitor and stop early
* Alpha-Spending Functions: O'Brien-Fleming boundaries
* Bootstrapping: Analyzing complex metrics (medians, percentiles, ratios)
* **Industry Spotlight:** Optimizely and the democratization of sequential testing

### [Chapter 12: Advanced Designs: Multi-Variant and Factorial Experiments](ch12_advanced_designs.md)
* Multi-Variant Testing (A/B/n) with ANOVA
* Factorial Experiments: Testing multiple features simultaneously
* Interaction Effects: Understanding combined feature impacts

### [Chapter 13: Evaluating Ranking Systems: Offline to Online Interleaving](ch13_ranking_experiments.md)
* Offline Metrics: nDCG, MRR, Precision@k
* Online Interleaving: Team-Draft and Balanced interleaving
* Statistical Analysis: Wilcoxon Signed-Rank Test
* **Industry Spotlight:** Bing's interleaving methodology for search quality

### [Chapter 14: Switchback and Geo-Experiments: Testing on Time and Space](ch14_switchback_and_geo_experiments.md)
* When Standard A/B Tests Fail: Network effects and spillover
* Switchback Experiments: Temporal randomization
* Geo-Experiments: Randomizing by geographic area
* **Industry Spotlight:** Uber and Lyft's marketplace experiments

### [Chapter 15: Multi-Armed Bandits: Balancing Exploration and Exploitation](ch15_multi_armed_bandits.md)
* Core Concepts: Epsilon-Greedy, Upper Confidence Bound (UCB), Thompson Sampling
* When to Use MABs: Short-term optimization vs. deep learning
* Implementation: Python examples with contextual bandits
* **Industry Spotlight:** Meta's use of MABs for optimization

---

## [Part IV: Online Experimentation for Machine Learning Systems](#part-iv-ml-experimentation)

This section focuses on ML-specific challenges: validating models through online experiments, using adaptive methods for hyperparameter tuning, leveraging experiment data for counterfactual learning, and building safe retraining pipelines.

### [Chapter 16: Testing Machine Learning Systems](ch16_testing_machine_learning_systems.md)
* Online vs. Offline Evaluation: The offline-online gap
* The Evaluation Funnel: Offline → Shadow → Canary → Full A/B test
* Testing ML Components: Features, model architectures, hyperparameters
* **Industry Spotlight:** Netflix's approach to recommendation algorithm testing

### [Chapter 17: Adaptive Experimentation for Model Optimization](ch17_adaptive_experimentation_for_model_optimization.md)
* The Hyperparameter Tuning Problem: Why grid/random search fail online
* Bayesian Optimization: Gaussian Processes and acquisition functions
* Implementation with Meta's Ax and BoTorch
* Integration with Experimentation Platforms

### [Chapter 18: Machine Learning from Experiment: Counterfactual Learning](ch18_using_experiment_data_for_model_training.md)
* The Value of Experiment Data: Clean causal signals
* Counterfactual Methods: IPS, Doubly Robust, Meta-Learners
* Instrumental Variables for contaminated data
* **Industry Spotlight:** Microsoft Research's counterfactual learning for ads

### [Chapter 19: Deploying Experiment-Trained Models: Safe Retraining Pipelines and Governance](ch19_experiment_data_safety_and_governance.md)
* The Retraining Problem: Temporal feedback loops
* Pipeline Design Patterns: Holdout sets, temporal buffers, validation gates
* Complete SafeExperimentRetrainer implementation
* Governance and Ethics: Fairness auditing, bias detection, transparency
* **Industry Spotlight:** LinkedIn's production ML retraining platform

---

## How to Use This Book

### For Data Engineers:
Start with **Part I** (Chapters 1-4) for statistical foundations, then focus on **Part II** (Chapters 5-10) for platform architecture and pipeline engineering. Use **Part III** (Chapters 11-15) as a reference when you encounter specific design challenges.

### For ML Engineers:
Read **Part I** for foundations, skim **Part II** for platform understanding, then dive deep into **Part IV** (Chapters 16-19) for ML-specific techniques. Reference **Part III** when you need specialized designs like interleaving or bandits.

### For Platform Architects:
Focus on **Chapter 5** for overall architecture, then **Chapters 6-9** for implementation details. Study the Industry Spotlights to learn from companies operating at scale.

### As a Reference:
Each chapter is designed to stand alone. Use the clickable table of contents to jump directly to topics relevant to your current challenge.

---

## Prerequisites

**Required Background:**
- Proficiency in Python (primary language for code examples)
- Basic SQL knowledge (for data pipeline examples)
- Familiarity with probability and statistics (undergraduate level)
- Experience with distributed systems (helpful but not required)

**Tools and Technologies:**
- Python 3.8+: SciPy, statsmodels, pandas, numpy
- Data pipeline tools: dbt, Airflow, Kafka
- Cloud platforms: AWS/GCP/Azure (examples are platform-agnostic)

**Companion Code Repository:**
All code examples are available at: [GitHub repository link]

---

## Acknowledgments

This book builds on decades of research and industry practice. We are grateful to:

- **Ron Kohavi, Diane Tang, and Ya Xu** for *Trustworthy Online Controlled Experiments*, the canonical reference that shaped modern experimentation practice
- The experimentation teams at Microsoft, Google, Netflix, LinkedIn, Meta, eBay, Uber, and Airbnb whose public talks, papers, and blog posts inform the Industry Spotlights
- The open-source community for tools like statsmodels, SciPy, dbt, and Airflow that make production experimentation accessible

---

*Let's build trustworthy experimentation systems together.*
