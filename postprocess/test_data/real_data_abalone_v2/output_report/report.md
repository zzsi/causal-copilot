# Causal Discovery of Abalone Biological Characteristics

## Background
The variable names in the provided dataset, "abalone.csv," are meaningful and related to the biological characteristics of abalones. Below is the detailed explanation of each variable, possible causal relations, and background domain knowledge relevant for causal discovery.

### 1. Detailed Explanation about the Variables

- **Sex**: Indicates the biological sex of the abalone (male, female, infant). Sex differences can influence physical characteristics and growth patterns.

- **Length**: Represents the length of the abalone in millimeters, directly measuring the size.

- **Diameter**: The diameter of the abalone in millimeters, another measure of size related to the overall body dimensions.

- **Height**: Indicates the height (thickness) of the abalone, measured in millimeters, contributing dimensions of size.

- **Whole weight**: The total weight (including shell and soft parts) of the abalone, measured in grams.

- **Shucked weight**: Weight of the abalone meat after removal from the shell, measured in grams.

- **Viscera weight**: Indication of the weight of the internal organs, providing insights into overall health and biological processes.

- **Shell weight**: The weight of the empty shell after removal of the soft body, measured in grams.

- **Rings**: Represents the number of rings on the shell, serving as an indicator of the abalone's age.

### 2. Possible Causal Relations among These Variables

- **Sex → Length, Diameter, Height**: Different sexes may exhibit variations in growth patterns affecting size.

- **Length, Diameter, Height → Whole Weight**: Larger dimensions lead to greater whole weight.

- **Whole Weight → Shucked Weight, Viscera Weight, Shell Weight**: An increase in whole weight likely affects the weights of its components.

- **Shell Weight → Rings**: Older abalones typically have heavier shells that may accumulate rings as they grow.

- **Age (Rings) → Length, Diameter, Height, Whole Weight**: Older abalones are generally larger across all dimensions and weights.

### 3. Other Background Domain Knowledge that May Be Helpful

- **Biological Development**: Factors such as temperature and food availability significantly influence abalone growth.

- **Reproductive Biology**: Knowledge of the reproductive cycle and its impact on growth can clarify size variations among sexes.

- **Ecological Factors**: The abalone's habitat affects growth and survival, impacting relationships among variables.

- **Market and Harvesting Practices**: Understanding harvesting methods may explain variations in size and weight data.

- **Physiological Factors**: Health aspects, including disease and competition, can influence size, weight, and shell characteristics.

## Dataset Descriptions
The following is a preview of our dataset:

| Sex | Length | Diameter | Height | Whole Weight | Shucked Weight | Viscera Weight | Shell Weight | Rings |
|-----|--------|----------|--------|--------------|----------------|----------------|--------------|-------|
| M   | 0.455  | 0.365    | 0.095  | 0.5140       | 0.2245         | 0.1010         | 0.150        | 15    |
| M   | 0.350  | 0.265    | 0.090  | 0.2255       | 0.0995         | 0.0485         | 0.070        | 7     |
| F   | 0.530  | 0.420    | 0.135  | 0.6770       | 0.2565         | 0.1415         | 0.210        | 9     |
| M   | 0.440  | 0.365    | 0.125  | 0.5160       | 0.2155         | 0.1140         | 0.155        | 10    |
| I   | 0.330  | 0.255    | 0.080  | 0.2050       | 0.0895         | 0.0395         | 0.055        | 7     |

### Data Properties and Exploratory Data Analysis Results

- **Sample Size**: 4177
- **Features**: 9 (mixture of categorical and numerical)
- **Data Quality**: No missing values
- **Statistical Properties**:
  - Linearity: Non-linear relationships observed
  - Gaussian Errors: Errors do not follow a Gaussian distribution
  - Heterogeneity: Dataset is homogeneous

### Distribution Analysis
![Distribution Graph](/postprocess/test_data/real_data_abalone_v2/output_graph/eda_dist.jpg)

#### Analysis of the Distributions

1. **Slight Left Skew Distributed Variables:**
   - **Length**: Mean (0.52) > Median (0.55) suggests slight left skew.

2. **Slight Right Skew Distributed Variables:**
   - **Whole Weight**: Mean (0.83) > Median (0.80) - slight right skew.
   - **Rings**, **Shucked Weight**, **Viscera Weight**, **Shell Weight**: All exhibit slight right skew, as their means exceed their medians.

3. **Symmetrically Distributed Variables:**
   - **Height**: Mean (0.14) = Median (0.14) indicates symmetry.
   - **Diameter**: Mean (0.41) ~ Median (0.42) - fairly symmetric.

### Categorical Features:
- **Sex**: The categorical variable with counts for three categories (M, F, I) can be analyzed for distributions.

This categorization identifies patterns and potential outliers, informing subsequent analyses.

### Correlation Analysis
![Scatter Plot](/postprocess/test_data/real_data_abalone_v2/output_graph/eda_scat.jpg)
![Correlation Graph](/postprocess/test_data/real_data_abalone_v2/output_graph/eda_corr.jpg)

#### Strong Correlations ( \( r \geq 0.8 \) )
- E.g., **Diameter & Length (0.99)**, **Whole Weight & Length (0.93)** - indicates tightly linked relationships.

#### Moderate Correlations ( \( 0.5 \leq r < 0.8 \) )
- E.g., **Height & Length (0.83)**, **Rings with various dimensions (e.g., 0.56 with Length)** indicative of less robust relationships.

#### Weak Correlations ( \( r < 0.5 \) )
- No correlations fit into this category; the lowest correlation is **0.50**.

### Summary
The dataset reveals strong interdependencies amongst size and weight variables, indicating possible redundancy in information. Moderate correlations exist with Rings, suggesting the need for careful consideration in predictive modeling to avoid multicollinearity.

## Discovery Procedure

### Step-by-Step Causal Discovery Procedure

1. **Data Preprocessing**: Clean, normalize, and verify the dataset.

2. **Algorithm Selection**: 
   - Shortlisted: **PC**, **GES**, **CDNOD**
   - Chosen: **PC Algorithm** for efficiency and compatibility with dataset size.

3. **Hyperparameter Values Proposal**:
   - Selected with justification via LLM:
     - `alpha`: 0.01 (low false positives)
     - `indep_test`: Fisher’s Z test (for continuous data)
     - `uc_rule`: 0 (default assumption)
     - `uc_priority`: 2 (manage conflicts conservatively)

4. **Graph Tuning**: Validate models using bootstrapping and refine with LLM suggestions.

### Selected Parameters
```json
{
  "selected_algorithm": {
    "name": "PC",
    "justification": "Efficient for large dataset size in discovering causal relationships."
  },
  "hyperparameters": {
    "alpha": {
      "value": 0.01,
      "explanation": "Reduces false positives."
    },
    "indep_test": {
      "value": "fisherz",
      "explanation": "Efficient for continuous non-Gaussian data."
    },
    "uc_rule": {
      "value": 0,
      "explanation": "Assumes no faithfulness violations."
    },
    "uc_priority": {
      "value": 2,
      "explanation": "Conservative management of collider conflicts."
    }
  }
}
```

## Results Summary
The following are result graphs produced by our algorithm:

| <center> Initial Graph                                                                      | <center> Reliability                                                                                |
|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| ![Initial Graph](/postprocess/test_data/real_data_abalone_v2/output_graph/Initial_Graph.jpg) | ![Confidence Heatmap](/postprocess/test_data/real_data_abalone_v2/output_graph/confidence_heatmap.jpg) |

The result graph identifies relationships essential to understanding how these biological variables interconnect. The Length influences both Diameter and Viscera weight, indicating size interdependence.

### Graph Reliability Analysis
Confidence probabilities reveal:
- High confidence in certain edges (e.g., **Length → Diameter**: 0.96).
- Contrarily, low confidence in edges like **Diameter → Height** (0.09) suggests further investigation is warranted.

In conclusion, while strong causal associations are evident among key variables, certain relationships need further validation through empirical studies. This reliability analysis suggests the potential for refinement and further exploration regarding causal hypotheses.
