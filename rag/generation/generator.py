# rag/generation/generator.py

from openai import OpenAI
from typing import Optional


class Generator:
    """
    A wrapper for the OpenAI ChatCompletion API.
    """

    def __init__(self, model: str = "gpt-4o", system_prompt: str = "You are a helpful squash training assistant."):
        """
        Initialises the OpenAI client.

        Args:
            model (str): The name of the model to use (e.g., "gpt-4o").
            system_prompt (str): The system prompt to guide the model's behaviour.
        """
        self.client = OpenAI()

        self.model = model
        self.system_prompt = system_prompt

    def generate(self, user_prompt: str, temperature: Optional[float] = None) -> str:
        """
        Sends the user prompt to the OpenAI API and returns the model's response.

        Args:
            user_prompt (str): The fully-formed prompt to send to the model.
            temperature (Optional[float]): The sampling temperature to use. If None,
                                           the API's default will be used.

        Returns:
            str: The content of the assistant's message.
        """
        # Build the dictionary of parameters for the API call
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        }

        # Only add the temperature parameter if it's explicitly provided
        if temperature is not None:
            api_params["temperature"] = temperature

        # Use the dictionary to make the API call
        response = self.client.chat.completions.create(**api_params)

        return response.choices[0].message.content