# evaluation/evaluators/llm_judge.py

import os
import json
from pathlib import Path
from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai library not found. The LlmEvaluator will not be available.")
    OpenAI = None


class LlmEvaluator:
    """
    A reusable class to handle evaluations using an LLM as a judge.

    This class manages API calls, prompt formatting, response parsing, and caching
    to ensure consistent and cost-effective evaluations.
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview", prompts_dir: str | Path = "evaluation/prompts"):
        if OpenAI is None:
            raise ImportError("Please install the openai library: pip install openai")

        self.model = model_name
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.is_dir():
            raise FileNotFoundError(f"Prompts directory not found at: {self.prompts_dir}")

        # Initialize OpenAI client (expects OPENAI_API_KEY environment variable)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # Simple in-memory cache to avoid repeated API calls during a single run
        self.cache = {}

    def _evaluate(self, prompt_name: str, **kwargs) -> Dict[str, Any]:
        """
        Core evaluation method that formats a prompt, calls the LLM, and parses the JSON response.

        Args:
            prompt_name (str): The base name of the .txt prompt file in the prompts directory.
            **kwargs: The variables to be formatted into the prompt template.

        Returns:
            Dict[str, Any]: The parsed JSON object from the LLM's response.
        """
        # Create a unique key for caching based on prompt and its arguments
        cache_key = json.dumps((prompt_name, sorted(kwargs.items())))
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            prompt_path = self.prompts_dir / f"{prompt_name}.txt"
            prompt_template = prompt_path.read_text(encoding="utf-8")

            final_prompt = prompt_template.format(**kwargs)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.0,  # Set to 0 for deterministic evaluation
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content
            parsed_json = json.loads(response_content)

            # Cache the successful result
            self.cache[cache_key] = parsed_json
            return parsed_json

        except (FileNotFoundError, KeyError) as e:
            print(f"Error with prompt formatting or file: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"LLM Raw Response: {response_content}")
            return {"error": "JSONDecodeError", "raw_response": response_content}
        except Exception as e:
            print(f"An unexpected error occurred during LLM evaluation: {e}")
            return {"error": str(e)}

    # --- Public wrapper methods for specific evaluation tasks ---

    def evaluate_logical_coherence(self, query: str, generated_plan: str) -> Dict[str, Any]:
        """Evaluates the structural and logical coherence of a plan."""
        return self._evaluate(
            prompt_name="judge_logical_coherence",
            query=query,
            generated_plan=generated_plan
        )

    def evaluate_faithfulness(self, context: str, generated_plan: str) -> Dict[str, Any]:
        """Evaluates the faithfulness of a plan against its context."""
        return self._evaluate(
            prompt_name="judge_faithfulness",
            context=context,
            generated_plan=generated_plan
        )

    def check_constraint_alignment(self, human_rule: str, formal_rule: str) -> float:
        """
        Checks if a human-readable rule aligns with a formal rule.
        Returns a float score (1.0 for alignment, 0.0 for misalignment).
        """
        result = self._evaluate(
            prompt_name="judge_constraint_alignment",
            human_rule=human_rule,
            formal_rule=formal_rule
        )
        # Safely get the score, defaulting to 0.0 on error
        return result.get("alignment_score", 0.0)