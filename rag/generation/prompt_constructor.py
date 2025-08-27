# rag/generation/prompter_constructor.py

from pathlib import Path
from typing import List, Dict

# Import the robust parser to infer session type
from rag.parsers.user_query_parser import parse_type


class PromptConstructor:
    """
    Handles the creation of final prompt strings for the generator.
    """

    def __init__(self, template_dir: str | Path):
        """
        Initialises the Prompter.

        Args:
            template_dir (str | Path): The directory where prompt templates (.txt files) are stored.
        """
        self.template_dir = Path(template_dir)
        if not self.template_dir.is_dir():
            raise FileNotFoundError(f"Prompt template directory not found at: {self.template_dir}")

    def create_prompt(
            self,
            query: str,
            documents: List[Dict],
            default_session_type: str = "conditioned_game"
    ) -> str:
        """
        Builds the final prompt by inferring session type, constructing context,
        and filling the appropriate template.

        Args:
            query (str): The original user query.
            documents (List[Dict]): The list of documents retrieved from the knowledge base.
            default_session_type (str): A fallback session type if one cannot be inferred.

        Returns:
            str: The final, formatted prompt string.
        """
        session_type = self._infer_session_type(query) or default_session_type  #
        context = self._build_context_string(documents)  #

        prompt_path = self.template_dir / f"session_{session_type}.txt"  #
        try:
            prompt_template = prompt_path.read_text(encoding="utf-8")  #
        except FileNotFoundError:
            print(f"Warning: Prompt template for '{session_type}' not found. Using default.")
            prompt_path = self.template_dir / f"session_{default_session_type}.txt"
            prompt_template = prompt_path.read_text(encoding="utf-8")

        return prompt_template.format(question=query, context=context)  #

    def _infer_session_type(self, query: str) -> str | None:
        """Infers the session type from the query using the user desires parser."""
        return parse_type(query)  #

    def _build_context_string(self, documents: List[Dict], max_tokens: int = 3000) -> str:
        """
        Constructs a single context string from a list of documents.

        """
        context_parts = []
        total = 0
        for doc in documents:
            content = doc.get("contents", "")
            tokens = len(content.split())
            if total + tokens > max_tokens:
                break
            context_parts.append(content)
            total += tokens
        return "\n---\n".join(context_parts)  #