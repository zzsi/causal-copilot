import re
import json
from openai import OpenAI

class LLMClient:
    def __init__(self, args):
        self.client = OpenAI()
        self.model = "gpt-4o-mini"  # Default model

    def chat_completion(self, prompt, system_prompt="You are a helpful assistant.", json_response=False, model=None):
        """
        Send a chat completion request to the LLM.
        
        Args:
            prompt (str): The user prompt
            system_prompt (str): The system prompt
            json_response (bool): Whether to expect JSON response
        
        Returns:
            str: The model's response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": self.model if model is None else model,
            "messages": messages,
        }
        
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        messages = self.client.chat.completions.create(**kwargs).choices[0].message.content

        if json_response:
            return json.loads(messages)
        else:
            return messages
