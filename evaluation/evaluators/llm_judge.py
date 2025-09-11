# evaluation/evaluators/llm_judge.py
import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai library not found. Install with: pip install openai")
    OpenAI = None


class LlmEvaluator:
    """
    Thin wrapper around OpenAI Chat Completions in JSON mode.
    Loads prompt templates from evaluation/prompts/*.txt and enforces JSON output.
    Includes a small in-memory cache per run.
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview", prompts_dir: str | Path = "evaluation/prompts"):
        load_dotenv()  # make sure OPENAI_API_KEY in .env is available

        if OpenAI is None:
            raise ImportError("Please install the openai library: pip install openai")

        self.model = model_name
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.is_dir():
            raise FileNotFoundError(f"Prompts directory not found at: {self.prompts_dir}")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

        # OpenAI client
        self.client = OpenAI(api_key=api_key)

        # simple per-run cache
        self.cache: Dict[str, Any] = {}

    # --------------- core ---------------

    def _evaluate(self, prompt_name: str, **kwargs) -> Dict[str, Any]:
        """
        Load prompts/{prompt_name}.txt, format with kwargs, call the model in JSON mode, return parsed dict.
        """
        cache_key = json.dumps((prompt_name, sorted(kwargs.items())), ensure_ascii=False)
        if cache_key in self.cache:
            return self.cache[cache_key]

        path = self.prompts_dir / f"{prompt_name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        template = path.read_text(encoding="utf-8")
        final_prompt = template.format(**kwargs)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.0,  # deterministic
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            self.cache[cache_key] = parsed
            return parsed
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON decode error from LLM response: {e}\nRaw: {content}") from e
        except Exception as e:
            raise RuntimeError(f"LLM evaluation failed: {e}") from e

    # --------------- convenience wrappers ---------------

    def extract_signature(self, plan_text: str) -> Dict[str, Any]:
        """Extract a structural signature from a plan (exercises, rules, motifs, blocks, focus, level)."""
        return self._evaluate("extract_signature", plan_text=plan_text)

    def judge_pairwise(self, sig_json_A: str, sig_json_B: str) -> Dict[str, Any]:
        """
        Compare two signatures and return boolean facet differences + overall distinctness (0/1/2).
        Pass JSON strings for stable formatting with .format().
        """
        return self._evaluate("judge_pairwise", sig_json_A=sig_json_A, sig_json_B=sig_json_B)

    # Optional existing wrappers
    def evaluate_logical_coherence(self, query: str, generated_plan: str) -> Dict[str, Any]:
        return self._evaluate("judge_logical_coherence", query=query, generated_plan=generated_plan)

    def evaluate_faithfulness(self, context: str, generated_plan: str) -> Dict[str, Any]:
        return self._evaluate("judge_faithfulness", context=context, generated_plan=generated_plan)

    def check_constraint_alignment(self, human_rule: str, formal_rule: str) -> float:
        result = self._evaluate("judge_constraint_alignment", human_rule=human_rule, formal_rule=formal_rule)
        return result.get("alignment_score", 0.0)
