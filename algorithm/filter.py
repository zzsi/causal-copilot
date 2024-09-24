from openai import OpenAI
import json

class Filter(object):
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)

    def load_context(self, filename):
        with open(f"algorithm/context/{filename}.txt", "r") as f:
            return f.read()

    def create_prompt(self, data, statics_dict):
        columns = ', '.join(data.columns)
        stats = json.loads(statics_dict)
        
        algo_context = self.load_context("algo")
        score_function_context = self.load_context("score_function")
        independence_test_context = self.load_context("independence_test")
        
        prompt = f"""Given a dataset with the following properties:

1. Columns: {columns}
2. Statistics:
   - Data Type: {stats['Data Type']}
   - Missingness: {stats['Missingness']}
   - Linearity: {stats.get('Linearity', 'N/A')}
   - Gaussian Error: {stats.get('Gaussian Error', 'N/A')}
   - Stationary: {stats.get('Stationary', 'N/A')}

Please analyze the dataset characteristics and recommend suitable causal discovery algorithms. Use the following context information to inform your recommendations:

Algorithms:
{algo_context}

Score Functions:
{score_function_context}

Independence Tests:
{independence_test_context}

For each recommended algorithm, provide:
1. Algorithm name
2. Brief description
3. Justification for its suitability based on the dataset characteristics
4. Suggested independence test or score function (if applicable)
5. Key hyperparameters to consider

Present your recommendations in a structured JSON format, focusing on the most suitable algorithms given the dataset characteristics. Limit your recommendations to the top 3 most suitable algorithms.

Please structure your response like this example:

{{
  "algorithms": [
    {{
      "name": "Algorithm Name",
      "description": "Brief description of the algorithm.",
      "justification": "Explanation of why this algorithm is suitable for the given dataset.",
      "independence_test_or_score_function": "Suggested test or function, if applicable.",
      "hyperparameters": {{
        "Hyperparameter 1": "Description or suggested value",
        "Hyperparameter 2": "Description or suggested value"
      }}
    }},
    {{
      "name": "Another Algorithm Name",
      "description": "Brief description of another algorithm.",
      "justification": "Explanation of why this algorithm is also suitable.",
      "independence_test_or_score_function": "Suggested test or function, if applicable.",
      "hyperparameters": {{
        "Hyperparameter A": "Description or suggested value",
        "Hyperparameter B": "Description or suggested value"
      }}
    }},
    {{
      "name": "Third Algorithm Name",
      "description": "Brief description of a third algorithm.",
      "justification": "Explanation of why this algorithm is also suitable.",
      "independence_test_or_score_function": "Suggested test or function, if applicable.",
      "hyperparameters": {{
        "Hyperparameter X": "Description or suggested value",
        "Hyperparameter Y": "Description or suggested value"
      }}
    }}
  ]
}}

Please provide your recommendations following this JSON structure.
"""
        return prompt

    def parse_response(self, response):
        try:
            algo_candidates = json.loads(response)
            return {algo['name']: {
                'description': algo['description'],
                'justification': algo['justification'],
                'independence_test_or_score_function': algo['independence_test_or_score_function'],
                'hyperparameters': algo['hyperparameters']
            } for algo in algo_candidates['algorithms']}
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON response")
            return {}

    def forward(self, data, statics_dict):
        prompt = self.create_prompt(data, statics_dict)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a causal discovery expert. Provide your response in JSON format."},
                {"role": "user", "content": prompt}
            ]
            response_format="json",
        )
        
        output = response.choices[0].message.content
        algo_candidates = self.parse_response(output)
        
        return algo_candidates