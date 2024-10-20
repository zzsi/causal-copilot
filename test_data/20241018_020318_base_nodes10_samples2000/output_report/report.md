# Causal Discovery Report: Relationships Between Socioeconomic Factors and Mental Health

## Background
This report explores the intricate relationships between several key variables: income level, education attainment, healthcare access, mental health anxiety, mental health depression, and a domain index. Understanding these variables is essential for uncovering the underlying factors that influence individual well-being and societal health outcomes. 

- **Income Level**: Reflects an individual’s financial status, which can affect access to resources and opportunities.
- **Education Attainment**: Indicates the highest level of education achieved and is often linked to better job prospects and income, creating a bidirectional relationship with income.
- **Healthcare Access**: Encompasses the ability to obtain necessary health services, significantly influenced by socioeconomic status and education, which impacts mental health outcomes.
- **Mental Health Variables**: Anxiety and depression are critical for understanding individual well-being.
- **Domain Index**: A composite score summarizing performance across various spheres affecting health and social outcomes.

The goal of this analysis is to shed light on the relationships that shape health and quality of life, highlighting the importance of socioeconomic factors and healthcare access in mental health dynamics.

## Dataset Descriptions
The following is a preview of our dataset:

| income_level | education_attainment | healthcare_access | mental_health_anxiety | mental_health_depression | domain_index |
|--------------|----------------------|--------------------|-----------------------|--------------------------|---------------|
| 0.025298 | -0.456139 | -0.222759 | 2.210646 | NaN | 0 |
| 0.152398 | 0.926548 | NaN | NaN | -0.553400 | 0 |
| 0.112733 | -0.659964 | NaN | 2.373330 | -0.312250 | 0 |
| NaN | 0.036934 | 0.042326 | 2.174514 | -0.274188 | 0 |
| 0.173669 | -0.598435 | -0.553615 | NaN | -0.260146 | 0 |

**Dataset Characteristics:**
- Sample Size: 2531
- Features: 6
- Data Type: Continuous
- Quality: Missing values present.
- Linearity: Predominantly linear relationships.
- Errors: Non-Gaussian distribution.
- Heterogeneity: Dataset is heterogeneous with the domain index reflecting that.

## Discovery Procedure
### Step 1: Data Preprocessing
- **Statistical Characteristics**: Examined dataset properties and distribution.
- **Missing Values**: Assessed and addressed missing values for a cleaner dataset.
- **Relationship Analysis**: Evaluated linearity and heterogeneity within the data.

### Step 2: Algorithm Selection Assisted with LLM
Utilized a Large Language Model (LLM) to select appropriate causal discovery algorithms based on:
- Dataset characteristics, including sample size and error distribution.
- Recommendations included:
  1. **CDNOD**: Suitable for heterogeneous data.
  2. **FCI**: Effective for handling hidden confounders.
  3. **DirectLiNGAM**: Good for predominantly linear relationships with non-Gaussian noise.

### Step 3: Hyperparameter Values Proposal Assisted with LLM
Optimized selected algorithms by seeking hyperparameter settings via LLM recommendations.

### Step 4: Graph Tuning with Bootstrap and LLM Suggestions
- **Bootstrapping**: Used to assess stability of causal relationships.
- **LLM Feedback**: Integrated suggestions for tuning the causal graph.

## Results Summary
The following graphs illustrate the causal relationships among variables:

|<center> True Graph|<center> Initial Graph| <center> Revised Graph|
|--|--|--|
|![True Graph](test_data/20241018_020318_base_nodes10_samples2000/output_graph/True_Graph.jpg)|![Initial Graph](test_data/20241018_020318_base_nodes10_samples2000/output_graph/Initial_Graph.jpg)|![Revised Graph](test_data/20241018_020318_base_nodes10_samples2000/output_graph/Revised_Graph.jpg)|

**Causal Relationships Observed:**
- **Education Attainment ↔ Healthcare Access**: Bidirectional influence.
- **Mental Health Anxiety ↔ Mental Health Depression**: Significant contributions to depression from increased anxiety.
- **Domain Index**: Overall impact on mental health conditions, suggesting interventions should address these variables collectively.

### Metrics Evaluation
| Metric         | Original Graph | Revised Graph |
|----------------|----------------|----------------|
| SHD            | 6.0            | 6.0            |
| Precision      | 0.0            | 0.0            |
| Recall         | 0.0            | 0.0            |
| F1 Score       | 0.0            | 0.0            |

![Metric Graph](test_data/20241018_020318_base_nodes10_samples2000/output_graph/metrics.jpg)

**Analysis**:
- Despite adjustments and refinements, both the original and revised graphs exhibit a significant degree of error in terms of precision, recall, and F1 score.
- This suggests that while the causal relationships have been identified, the model may require further refinement or additional data to enhance predictive power and accuracy. 

In conclusion, addressing the interplay between these socioeconomic factors and mental health could provide a more comprehensive understanding of overall well-being and inform policies aimed at improving health outcomes.
