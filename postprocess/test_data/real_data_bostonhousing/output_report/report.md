# Causal Discovery Analysis of the Boston Housing Dataset

## Background
This report explores the interrelationships among various variables in the Boston Housing dataset, shedding light on factors that influence housing prices and neighborhood characteristics. The dataset comprises attributes like crime rate (CRIM), proportion of residential land (ZN), industrial land proportion (INDUS), proximity to the Charles River (CHAS), air pollution levels (NOX), average number of rooms (RM), age of housing stock (AGE), distance to employment centers (DIS), accessibility to highways (RAD), property tax rates (TAX), pupil-teacher ratios (PTRATIO), percentage of lower status individuals (LSTAT), and the median housing value (MEDV). Each variable plays a critical role in defining living conditions and market dynamics. Understanding these variables and their interconnections is essential for robust causal discovery analysis, providing insights into the interplay of socio-economic and environmental factors that shape the Boston housing market.

## Dataset Descriptions
The Boston Housing dataset comprises 506 samples and 14 features. Below is a preview of the dataset:

|   CRIM   |   ZN  |   INDUS  |   CHAS   |   NOX  |   RM   |   AGE   |   DIS   |   RAD  |   TAX   |   PTRATIO  |    B   |   LSTAT   |   MEDV  |
|----------|-------|----------|----------|--------|--------|---------|---------|--------|---------|------------|--------|-----------|---------|
|  0.00632 |  18.0 |   2.31   |   0.0    | 0.538  | 6.575  |  65.2   |  4.0900 |   1    |   296   |    15.3    | 396.90 |   4.98    |   24.0  |
|  0.02731 |  0.0  |   7.07   |   0.0    | 0.469  | 6.421  |  78.9   |  4.9671 |   2    |   242   |    17.8    | 396.90 |   9.14    |   21.6  |
| 0.02729  |  0.0  |   7.07   |   0.0    | 0.469  | 7.185  |  61.1   |  4.9671 |   2    |   242   |    17.8    | 392.83 |   4.03    |   34.7  |
| 0.03237  |  0.0  |   2.18   |   0.0    | 0.458  | 6.998  |  45.8   |  6.0622 |   3    |   222   |    18.7    | 394.63 |   2.94    |   33.4  |
| 0.06905  |  0.0  |   2.18   |   0.0    | 0.458  | 7.147  |  54.2   |  6.0622 |   3    |   222   |    18.7    | 396.90 |   NaN     |   36.2  |

### Dataset Characteristics
- **Sample Size**: 506
- **Number of Features**: 14
- **Data Type**: Continuous
- **Data Quality**: Contains missing values
- **Statistical Properties**:
  - Relationships between variables are not predominantly linear.
  - Errors in the data do not follow a Gaussian distribution.
  - The dataset is homogeneous.

### Implications for Analysis
1. Non-linear modeling techniques may be more appropriate.
2. Robust statistical methods or transformations might be necessary.
3. Imputation techniques should be considered during preprocessing.

## Discovery Procedure
### Causal Discovery Procedure Description

1. **Data Preprocessing**:
   - Clean the dataset by addressing any missing values through imputation. Assess statistical properties, including distribution, correlations, and linearity to ensure suitability for causal discovery.

2. **Algorithm Selection**:
   - Utilize a Large Language Model (LLM) to help select appropriate causal discovery algorithms. Based on dataset characteristics, we consider:
     - **CDNOD**: Not suitable due to lack of heterogeneity.
     - **FCI**: Effective for hidden confounders and identifying causal relationships through a Partial Ancestral Graph (PAG).
     - **NOTEARS**: Effective but computationally intensive.
   - We select the **FCI** algorithm for its robustness against hidden confounders and non-homogeneous structures.

3. **Hyperparameter Values Proposal**:
   - Proposed hyperparameters for the **FCI** algorithm:
   ```json
   {
     "algorithm": "FCI",
     "hyperparameters": {
       "alpha": {
         "value": 0.1,
         "explanation": "A lenient significance level helps capture more relationships given the sample size."
       },
       "indep_test": {
         "value": "fisherz",
         "explanation": "Fisher's Z test is suitable for evaluating conditional independence in continuous datasets."
       },
       "depth": {
         "value": -1,
         "explanation": "Allows unlimited depth for adjacency search to uncover complex causal pathways."
       }
     }
   }
   ```

4. **Graph Tuning**:
   - Apply the chosen **FCI** algorithm to derive the initial causal graph. Adjust the graph based on statistical tests and validate results using bootstrap methods. Utilize LLM insights for fine-tuning and interpreting discovered relationships.

In summary, our causal discovery procedure systematically employs data preprocessing, algorithm selection, hyperparameter tuning, and graph adjustment to uncover the causal structures in the Boston Housing dataset.

## Results Summary
The following graphs were produced by our analysis. The initial graph represents the first attempt, while the revised graph is pruned with bootstrapping and LLM suggestions.

| <center> Initial Graph | <center> Revised Graph |
|------------------------|------------------------|
| ![Initial Graph](t/postprocess/test_data/real_data_bostonhousing/output_graph/Initial_Graph.jpg) | ![Revised Graph](/postprocess/test_data/real_data_bostonhousing/output_graph/Revised_Graph.jpg) |

### Causal Relationships Insights
- The variable **ZN** influences **PTRATIO**, highlighting how zoning regulations affect pupil-teacher ratios.
- **INDUS** significantly impacts both **DIS** and **TAX**, suggesting industrial areas have lower access to amenities and higher taxes.
- **NOX** has a negative influence on **AGE** and **DIS**, indicating that higher pollution correlates with older housing and reduced accessibility.
- **AGE** affects **LSTAT**, showing older homes associate with a higher percentage of lower socioeconomic status residents.
- **DIS** also influences both **ZN** and **AGE**, indicating a bidirectional relationship.
- **RAD** affects **TAX**, illustrating how access to highways influences property tax rates.

These insights emphasize the complex interdependencies among these variables, reflecting the multifaceted nature of urban development and societal factors affecting housing markets.
