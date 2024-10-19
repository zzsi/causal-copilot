# Causal Discovery of Health-Related Variables

## Background
The purpose of this report is to explore and analyze the intricate relationships among five significant health-related variables: physical activity, dietary fat, fiber intake, blood pressure, and cholesterol. Understanding these variables is crucial, as each plays a vital role in influencing overall health and well-being. Physical activity refers to the amount of exercise an individual engages in, which can enhance cardiovascular fitness and impact both blood pressure and cholesterol levels. Dietary fat denotes the types and quantities of fats consumed, significantly affecting cholesterol levels and potentially impacting blood pressure as well. Fiber intake, representing the amount of dietary fiber consumed, is known to promote digestive health and can lower cholesterol levels and blood pressure. Blood pressure itself measures the pressure exerted by circulating blood on the arterial walls, with hypertension raising the risk of serious health issues. Meanwhile, cholesterol levels indicate the amount of cholesterol present in the blood, influenced by dietary choices and lifestyle factors. There are potential causal relationships among these variables, such as increased physical activity leading to lower blood pressure and healthier cholesterol levels, while dietary factors like fat and fiber intake can also modulate both cholesterol and blood pressure. By investigating these interconnections, the report aims to uncover deeper insights into how lifestyle choices impact health outcomes, providing a foundation for further research and causal discovery analysis.

## Dataset Descriptions
The following is a preview of our dataset:

| physical_activity | dietary_fat | fiber_intake | blood_pressure | cholesterol |
|-------------------|-------------|--------------|----------------|-------------|
| 0.280218          | 0.344066    | -0.203135    | 0.927715       | -0.676094   |
| -0.923927        | NaN         | NaN          | 0.924791       | NaN         |
| 0.608669          | 0.509279    | -0.023469    | NaN            | -0.653636   |
| 0.434224          | 0.318093    | -0.015076    | 0.855652       | -0.550986   |
| -0.053118         | 0.032296    | 0.083688     | 1.022114       | NaN         |

- **Sample Size**: 5000
- **Data Type**: Continuous variables
- **Data Quality**: Missing values present
- **Linearity**: Predominantly linear relationships
- **Gaussian Errors**: Errors do not follow a Gaussian distribution
- **Heterogeneity**: The dataset is not heterogeneous

### Implications for Analysis
1. Robust statistical methods or transformations might be necessary.
2. Imputation techniques should be considered during preprocessing.

## Discovery Procedure
### Step 1: Data Preprocessing
1. **Data Cleaning**: Address missing values using imputation techniques.
2. **Statistical Analysis**: Evaluate data relationships and error distributions.

### Step 2: Algorithm Selection
1. **Dataset Characteristics**: Understand relationships to inform algorithm selection.
2. **Candidate Algorithms**:
   - **DirectLiNGAM**: For linear causal structures with non-Gaussian errors.
   - **GES**: For efficiency with larger datasets.
   - **NOTEARS**: Not selected as primary method.
3. **Final Algorithm Selection**: **DirectLiNGAM**.

### Step 3: Hyperparameter Values Proposal
The hyperparameters for **DirectLiNGAM** are as follows:

```json
{
  "algorithm": "DirectLiNGAM",
  "hyperparameters": {
    "measure": {
      "value": "pwling",
      "explanation": "Using 'pwling' for pairwise likelihood-based measure is suitable as it aligns with the objective to evaluate independence in a dataset with non-Gaussian errors."
    },
    "random_state": {
      "value": "fixed_seed",
      "explanation": "Setting the random state to a fixed integer (e.g., 42) ensures reproducibility of results, which is essential in scientific investigations."
    },
    "prior_knowledge": {
      "value": null,
      "explanation": "Setting prior knowledge to null means no initial assumptions about causal relationships, allowing the algorithm to discover patterns purely from the data."
    },
    "apply_prior_knowledge_softly": {
      "value": false,
      "explanation": "Setting this to false means that any prior knowledge will be applied strictly."
    }
  }
}
```

### Step 4: Graph Tuning
1. **Graph Construction**: Create a directed acyclic graph (DAG).
2. **Bootstrapping**: Iterate for robustness of identified causal relationships.
3. **Model Refinement**: Adjust based on statistical insights and domain knowledge.

## Results Summary
The causal relationships identified among the variables are laid out in graphs generated from the algorithm. The initial graph and the refined graph after bootstrapping and LLM suggestions are displayed below.

|<center> True Graph |<center> Initial Graph| <center> Revised Graph|
|--|--|--|
| ![True Graph](test_data/20241018_020318_base_nodes10_samples2000/output_graph/True_Graph.jpg)| ![Initial Graph](test_data/20241018_020318_base_nodes10_samples2000/output_graph/Initial_Graph.jpg)| ![Revised Graph](test_data/20241018_020318_base_nodes10_samples2000/output_graph/Revised_Graph.jpg)|

### Analysis of Result Graphs
The dietary fat has a causal influence on physical activity, suggesting higher dietary fat impacts exercise levels. Fiber intake also positively influences physical activity. Furthermore, both blood pressure and cholesterol levels are contingent on physical activity, illustrating exercise's role in overall health management. The direct connection between cholesterol and blood pressure emphasizes the intricate relationships that could dictate cardiovascular health.

### Metrics Evaluation
Here's a comparison of metrics for the original and revised graph:

| Metric   | Original Graph | Revised Graph |
|----------|----------------|---------------|
| SHD      | 11.0           | 10.0          |
| Precision | 0.125          | 0.0           |
| Recall    | 0.2            | 0.0           |
| F1 Score  | 0.1538         | 0.0           |

![Metric Graph](test_data/20241018_020318_base_nodes10_samples2000/output_graph/metrics.jpg)

### Analysis of Metrics
The metrics indicate a slight improvement in the structural Hamming distance (SHD) with the revised graph, suggesting fewer erroneous edges. However, both precision and recall dropped to zero in the revised graph, implying that the refined graph might not have captured any true positive causal relationships despite a lower number of false positives, resulting in an F1 score of zero. This may imply that further refinement or adjustment is needed to achieve a more accurate representation of the causal structure among the variables. 

## Conclusion
This structured analysis demonstrates the complex relationships between physical activity, dietary habits, and health metrics such as blood pressure and cholesterol levels. While the methodological approach has provided insights, further refinement and analysis will be necessary to enhance the accuracy of causal relationships identified.
