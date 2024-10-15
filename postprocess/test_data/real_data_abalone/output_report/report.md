# Causal Discovery in Abalone Characteristics

## Background
The purpose of this report is to explore and analyze the relationships among various biological and physical characteristics of abalones, specifically focusing on the following variables: **Sex**, **Length**, **Diameter**, **Height**, **Whole weight**, **Shucked weight**, **Viscera weight**, **Shell weight**, and **Rings**. Understanding these variables is crucial for assessing growth patterns, health, and maturity of abalones. 

The categorical variable **Sex** signifies the biological sex of the abalone, which may impact growth and size. The continuous variables **Length**, **Diameter**, and **Height** measure the abalone's size and are positively correlated with weight metrics: **Whole weight**, **Shucked weight**, **Viscera weight**, and **Shell weight**. The number of **Rings** observed on the shell serves as an indicator of age, reflecting growth and potential differences in weight and size. 

Possible causal relationships suggest that sex may influence growth rates, while growth metrics are interrelated with various weight measures. Understanding these complex interactions is essential for implementing effective conservation strategies, managing harvesting practices, and enhancing the economic value of abalones. Various environmental factors also play a vital role in these relationships, indicating the necessity for a comprehensive approach to causal discovery in this context.

## Dataset Descriptions
Here is a preview of our dataset:

|  Sex | Length | Diameter | Height | Whole weight | Shucked weight | Viscera weight | Shell weight | Rings |
|------|--------|----------|--------|--------------|----------------|----------------|--------------|-------|
|   M  | 0.455  | 0.365    | 0.095  | 0.5140       | 0.2245         | 0.1010         | 0.150        |  15   |
|   M  | 0.350  | 0.265    | 0.090  | 0.2255       | 0.0995         | 0.0485         | 0.070        |  7    |
|   F  | 0.530  | 0.420    | 0.135  | 0.6770       | 0.2565         | 0.1415         | 0.210        |  9    |
|   M  | 0.440  | 0.365    | 0.125  | 0.5160       | 0.2155         | 0.1140         | 0.155        |  10   |
|   I  | 0.330  | 0.255    | 0.080  | 0.2050       | 0.0895         | 0.0395         | 0.055        |  7    |

### Dataset Characteristics:
- **Sample Size**: 4177
- **Features**: 9
- **Data Type**: Mixture
- **Data Quality**: No missing values

### Statistical Properties:
- **Linearity**: The relationships between variables are not predominantly linear.
- **Gaussian Errors**: Errors in the data do not follow a Gaussian distribution.
- **Heterogeneity**: This dataset is not heterogeneous.

### Implications for Analysis:
1. Non-linear modeling techniques may be more appropriate.
2. Robust statistical methods or transformations might be necessary.

## Discovery Procedure

### Causal Discovery Procedure

1. **Data Preprocessing**:
   - Load the dataset from the specified location (`test_data/real_data_abalone/abalone.csv`).
   - Inspect data types and identify both categorical and continuous variables.
   - Check for any missing values (none detected).

2. **Statistical Analysis**:
   - Analyze the statistical properties of the variables:
     - Identify relationships that are primarily non-linear.
     - Confirm the absence of Gaussian errors and note the homogeneity of the dataset.

3. **Algorithm Selection**:
   - Utilize a language model (LLM) to select an appropriate causal discovery algorithm based on the dataset’s characteristics and relevant domain knowledge.
   - Select the **PC algorithm** due to its efficiency with large datasets and capability to test conditional independencies.

4. **Hyperparameter Selection**:
   - Define necessary hyperparameters for the PC algorithm: `alpha`, `indep_test`, and `depth`.

5. **Suggested Hyperparameters**:
   Here are the recommended values for the hyperparameters, based on the dataset characteristics:

```json
{
  "algorithm": "PC",
  "hyperparameters": {
    "alpha": {
      "value": 0.01,
      "explanation": "Using a lower significance level (0.01) helps reduce false positives, especially critical with a large sample size."
    },
    "indep_test": {
      "value": "fisherz",
      "explanation": "Fisher's Z test is effective for assessing conditional independence given the continuous and non-linear nature of variables."
    },
    "depth": {
      "value": -1,
      "explanation": "The default setting of -1 allows exploration of all potential causal structures within the dataset."
    }
  }
}
```

### Step-by-Step Summary:
1. **Preprocessing and Analysis**: Load and analyze the data for quality and relationships.
2. **Algorithm Selection**: Choose PC as the best-suited algorithm.
3. **Hyperparameter Tuning**: Adjust key hyperparameters to optimize the causal discovery outcome.

## Results Summary
The following are result graphs produced by our algorithm. The initial graph shows the outcomes of the first analysis attempt, while the revised graph reflects the outcomes post-pruning with bootstrapping and suggestions from the language model.

| <center> Initial Graph | <center> Revised Graph |
|--|--|
| ![Initial Graph](/postprocess/test_data/real_data_abalone/output_graph/Initial_Graph.jpg) | ![Revised Graph](/postprocess/test_data/real_data_abalone/output_graph/Revised_Graph.jpg) |

### Observations:
The causal graph illustrates the relationships among the variables effectively. Notably:

- **Whole weight** is found to cause **Shucked weight**, emphasizing that fluctuations in overall mass directly affect edible weight.
- **Viscera weight** influences **Whole weight**, suggesting that increases in internal organ weight contribute significantly to the organism's overall mass.

These findings highlight the interconnected nature of the variables, providing crucial insights for biological research and resource management strategies.
