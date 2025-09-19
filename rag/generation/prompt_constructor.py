# rag/generation/prompter_constructor.py

from pathlib import Path
from typing import List, Dict
from collections import Counter


# Import the robust parser to infer session type
from rag.parsers.user_query_parser import parse_type

from pathlib import Path
from typing import List, Dict


class PromptConstructor_v2:
    """
    A simple prompt constructor that reads a single, pre-made prompt template file.
    """

    def __init__(self, template_path: str | Path):
        """
        Initialises the constructor by loading a specific prompt template from a file.

        Args:
            template_path (str | Path): The exact file path to the prompt template (.txt file).
        """
        self.template_path = Path(template_path)
        if not self.template_path.is_file():
            raise FileNotFoundError(f"Prompt template file not found at: {self.template_path}")
        # Read the template once during initialization for efficiency
        self.template = self.template_path.read_text(encoding="utf-8")

    def create_prompt(self, query: str, documents: List[Dict]) -> str:
        """
        Builds the final prompt by filling the pre-loaded template with context and a query.

        Args:
            query (str): The original user query.
            documents (List[Dict]): The list of documents to be used as context.

        Returns:
            str: The final, formatted prompt string.
        """
        # Build the context string using the helper method
        context_str = self._build_context_string(documents)

        # Format the pre-loaded template.
        # This assumes your light_touch.txt file uses {query} and {context} placeholders.
        return self.template.format(query=query, context=context_str)

    def _build_context_string(self, documents: List[Dict], max_tokens: int = 3000) -> str:
        """
        Constructs a single context string from a list of documents, respecting a token limit.
        This helper function is adapted from your original PromptConstructor.
        """
        context_parts = []
        total_tokens = 0
        for doc in documents:
            content = doc.get("contents", "")
            # A simple way to estimate tokens is by splitting by space
            tokens = len(content.split())
            if total_tokens + tokens > max_tokens:
                break
            context_parts.append(content)
            total_tokens += tokens
        return "\n\n---\n\n".join(context_parts)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
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


# In rag/generation/prompt_constructor.py
from collections import Counter


class DynamicPromptConstructor:
    def __init__(self):
        self.base_template = """You are an elite squash coach who is a machine for creating session plans.
        You will design a squash training session based on the provided context.

        **CRITICAL INSTRUCTIONS:**
        1.  **Stick to the Context:** You must ONLY use the exact drill names, point counts, and durations found in the provided 'Context Documents'. Do not invent or modify them.
        2.  **Follow the Format:** Adhere strictly to the 'OUTPUT FORMAT' structure provided below. Do not add extra notes, objectives, or commentary.
        3.  **Be Concise:** Generate only the session plan.

        ---------------------
        **CONTEXT DOCUMENTS:**
        {context}
        ---------------------

        **USER REQUEST:** {question}

        ---------------------

        **OUTPUT FORMAT:**

        Duration: {duration} min
        Session Focus: {focus}

        ### Warm-up ###
        {warmup_section}

        ### Activity Block 1 ###
        {activity_section_1}

        ### Activity Block 2 ###
        {activity_section_2}

        ... (Continue with more Activity Blocks as needed to fill the session time) ...

        -------------------------------------------------------------------------------
        End of session.
        """

    def _analyse_docs(self, retrieved_docs: list[dict]) -> dict:
        """Analyzes metadata from retrieved docs to find common themes."""
        if not retrieved_docs:
            return {}

        # Use Counter to find the most common value for key fields
        durations = Counter(doc['meta']['duration'] for doc in retrieved_docs if 'duration' in doc.get('meta', {}))
        levels = Counter(doc['meta']['recommended_squash_level'] for doc in retrieved_docs if
                         'recommended_squash_level' in doc.get('meta', {}))

        analysis = {}
        if durations:
            analysis['duration'] = durations.most_common(1)[0][0]
        if levels:
            analysis['level'] = levels.most_common(1)[0][0]

        # You can add more complex analysis here (e.g., for shot types)
        return analysis

    def create_prompt(self, query: str, retrieved_docs: list[dict]) -> str:
        """Builds the final prompt dynamically."""

        analysis = self._analyse_docs(retrieved_docs)

        # We can create a simple focus based on the query. More advanced logic could go here.
        # For now, we'll let the LLM infer it based on the query and context.
        session_focus = "Inferred from User Request and Context"  # Placeholder for LLM

        # Determine session duration from analysis, fallback to a default if needed.
        session_duration = analysis.get('duration', 45)  # Default to 45 mins

        # Format the context from documents
        formatted_context = "\n\n".join(
            [f"--- Document {i + 1} (ID: {doc.get('id', 'N/A')}) ---\n" + doc.get("contents", "") for i, doc in
             enumerate(retrieved_docs)]
        )

        return self.base_template.format(
            context=formatted_context,
            question=query,
            duration=session_duration,
            focus=session_focus,
            # We leave the actual content sections blank for the LLM to fill
            warmup_section="<LLM to generate based on context>",
            activity_section_1="<LLM to generate based on context>",
            activity_section_2="<LLM to generate based on context>"
        )