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
        prompt_template = self.load_context("algo_select_prompt")

        replacements = {
            "[COLUMNS]": columns,
            "[DATA_TYPE]": stats['Data Type'],
            "[MISSINGNESS]": str(stats['Missingness']),
            "[LINEARITY]": str(stats.get('Linearity', 'N/A')),
            "[GAUSSIAN_ERROR]": str(stats.get('Gaussian Error', 'N/A')),
            "[STATIONARY]": str(stats.get('Stationary', 'N/A')),
            "[ALGO_CONTEXT]": algo_context,
        }

        for placeholder, value in replacements.items():
            prompt_template = prompt_template.replace(placeholder, value)

        return prompt_template

    def parse_response(self, response):
        try:
            algo_candidates = json.loads(response)
            return {algo['name']: {
                'description': algo['description'],
                'justification': algo['justification']
            } for algo in algo_candidates['algorithms']}
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON response")
            return {}

    def forward(self, data, statics_dict):
        prompt = self.create_prompt(data, statics_dict)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a causal discovery expert. Provide your response in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        output = response.choices[0].message.content
        algo_candidates = self.parse_response(output)

        return algo_candidates