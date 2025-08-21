# src/grammar_tools/dedup/hash_dedup.py
"""
Performs exact deduplication by hashing session plans or their rendered text.
This version uses a canonical representation for session plans to ensure
order-independent deduplication.
"""
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

# Import the new canonicalizer function
from .canonical_form import to_canonical_plan


class HashDeduper:
    def __init__(self, mode: str = "plan"):
        if mode not in ["plan", "text"]:
            raise ValueError("HashDeduper mode must be 'plan' or 'text'")
        self.mode = mode
        self.seen_hashes = set()

    def is_duplicate(
            self,
            plan: Optional[Dict[str, Any]] = None,
            text: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Checks if a session is a duplicate based on its hash. If the mode is 'plan',
        it first converts the plan to a canonical form.
        """
        session_hash = ""

        if self.mode == "plan":
            if not plan:
                return False, ""

            # Convert the plan to its canonical form before hashing
            canonical_plan = to_canonical_plan(plan)
            # Use sort_keys=True to ensure the JSON string is always consistent
            content_string = json.dumps(canonical_plan, sort_keys=True)
            session_hash = hashlib.sha256(content_string.encode()).hexdigest()

        elif self.mode == "text":
            if text is None:
                return False, ""
            content_string = text
            session_hash = hashlib.sha256(content_string.encode()).hexdigest()

        if session_hash and session_hash in self.seen_hashes:
            return True, session_hash

        return False, session_hash

    def mark_seen(self, session_hash: str):
        """Adds a hash to the set of seen hashes."""
        if session_hash:
            self.seen_hashes.add(session_hash)