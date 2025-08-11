# src/grammar_tools/parser.py
# Minimal recursive-descent parser for the Sequence DSL defined in grammar/dsl/sequence.peg.
# Parse ONE line at a time; combine multiple lines by concatenating their step lists.

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

# Closed set of actor labels (normalized when validating)
ALLOWED_ACTORS = {
    "A",
    "B",
    "initiating player",
    "whoever chooses",
    "opponent",                   # generic "the other player"
    "opponent of cross",          # dependent on prior 'cross'
    "opponent of initiating player",
}

# ---------- Public API ----------

def parse_rules_sequence(seq: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Parse a rules.sequence value (string or list of strings) into a flat list of Step nodes.
    Each node is a dict with a 'type' key: Action | Choice | Repeat | Optional | Restart.

    Examples that parse:
      "boast (A) → cross (B) → drive (A)"
      "( straight drop (B) | straight drop (A) )* # zero or more"
      "optional: ( extra drive (A) | extra drive (B) )"
      "drive (A) # mandatory"
      "→ restart pattern at next boast"   # leading arrow OK; stripped
    """
    if isinstance(seq, str):
        lines = [seq]
    elif isinstance(seq, list):
        lines = list(seq)
    else:
        raise TypeError("rules.sequence must be a string or list of strings")

    ast_steps: List[Dict[str, Any]] = []
    for i, raw_line in enumerate(lines, start=1):
        line = _normalize_leading_arrow(raw_line)
        if _looks_like_if_else(line):
            raise ParseError(i, 1, "IF/ELSE is not part of the DSL; put branching in rules.description.")
        parser = _LineParser(line, line_no=i)
        steps = parser.parse_sequence()
        ast_steps.extend(steps)
    return ast_steps


# ---------- Errors & utilities ----------

class ParseError(Exception):
    def __init__(self, line: int, col: int, msg: str):
        super().__init__(f"Parse error at line {line}, col {col}: {msg}")
        self.line = line
        self.col = col
        self.msg = msg

def _normalize_leading_arrow(s: str) -> str:
    """Strip a leading arrow '→' or '->' and following spaces (authoring convenience)."""
    t = s.lstrip()
    if t.startswith("→"):
        return t[1:].lstrip()
    if t.startswith("->"):
        return t[2:].lstrip()
    return s

def _looks_like_if_else(s: str) -> bool:
    t = s.lstrip()
    return t.upper().startswith("IF ") or t.upper().startswith("ELSE")


# ---------- Core parser (single line) ----------

@dataclass
class _LineParser:
    text: str
    line_no: int

    def __init__(self, text: str, line_no: int = 1):
        self.text = text
        self.line_no = line_no
        self.i = 0  # cursor

    # entry point
    def parse_sequence(self) -> List[Dict[str, Any]]:
        self._ws()
        if self._eol():
            return []
        steps = [self._parse_step()]
        self._step_comment_opt()
        while True:
            saved = self.i
            self._ws()
            if not self._arrow_opt():
                self.i = saved
                break
            self._ws()
            steps.append(self._parse_step())
            self._step_comment_opt()
        self._trailing_opt()
        self._ws()
        # must be at end
        if not self._eol():
            col = self.i + 1
            snippet = self.text[self.i:self.i+15]
            raise ParseError(self.line_no, col, f"unexpected text: {snippet!r}")
        return steps

    # Step := Restart / Optional / Repeat / Choice / Action
    def _parse_step(self) -> Dict[str, Any]:
        self._ws()
        # Try Restart (must start with 'restart')
        if self._peek_word_ci("restart"):
            return self._parse_restart()
        # Try Optional
        if self._peek_word_ci("optional:"):
            return self._parse_optional()
        # Try Repeat (Group '*')
        if self._peek("("):
            saved = self.i
            group_node = self._parse_group()
            self._ws()
            if self._match("*"):
                return {"type": "Repeat", "body": group_node}
            # If no '*', then the group is either Choice or single Action
            if group_node["type"] == "Choice":
                return group_node
            if group_node["type"] == "Action":
                return group_node
            # Shouldn't happen; fall back to saved position for safety
            self.i = saved
        # Fallback: Action
        return self._parse_action()

    # Restart := 'restart' WS FreeText?
    def _parse_restart(self) -> Dict[str, Any]:
        self._expect_word_ci("restart")
        self._ws()
        # consume everything else (excluding newline) as text until end of line.
        text = self.text[self.i:].rstrip()
        # Move cursor to end of line so sequence parser can finish.
        self.i = len(self.text)
        node = {"type": "Restart"}
        if text:
            node["text"] = text
        return node

    # Optional := 'optional:' WS (Action | Group)
    def _parse_optional(self) -> Dict[str, Any]:
        self._expect_word_ci("optional:")
        self._ws()
        if self._peek("("):
            body = self._parse_group()
        else:
            body = self._parse_action()
        return {"type": "Optional", "body": body}

    # Group := '(' WS (ChoiceInner | Action) WS ')'
    # ChoiceInner := Action (WS '|' WS Action)+
    def _parse_group(self) -> Dict[str, Any]:
        self._expect("(")
        self._ws()
        # try choice: Action ( '|' Action )+
        first = self._parse_action()
        self._ws()
        if self._match("|"):
            options = [first]
            while True:
                self._ws()
                options.append(self._parse_action())
                self._ws()
                if not self._match("|"):
                    break
            self._ws()
            self._expect(")")
            return {"type": "Choice", "options": options}
        # else single action in parens
        self._expect(")")
        return first  # Action node

    # Action := Name Actor? InlineComment?
    def _parse_action(self) -> Dict[str, Any]:
        name = self._parse_name()
        actor = None
        # Actor := '(' ActorName ')'
        saved = self.i
        self._ws()
        if self._match("("):
            self._ws()
            actor = self._parse_actor_name()
            self._ws()
            self._expect(")")
        else:
            self.i = saved
        # InlineComment := WS? '#' ...
        self._inline_comment_opt()
        return {"type": "Action", "name": name, **({"actor": actor} if actor else {})}

    # Name := Quoted | Word (WS Word)*
    def _parse_name(self) -> str:
        self._ws()
        if self._match('"'):
            buf = []
            while not self._eol():
                # support \" escapes
                if self._peek('\\"'):
                    self.i += 2
                    buf.append('"')
                    continue
                ch = self._peek_char()
                if ch == '"':
                    self.i += 1  # consume closing quote
                    break
                if ch in ("\n", "\r"):
                    col = self.i + 1
                    raise ParseError(self.line_no, col, "unterminated quoted string")
                buf.append(ch)
                self.i += 1
            return "".join(buf).strip()
        # Word sequence
        w = self._parse_word()
        words = [w]
        saved = self.i
        while True:
            try:
                self._ws1()
                words.append(self._parse_word())
                saved = self.i
            except ParseError:
                self.i = saved
                break
        return " ".join(words).strip()

    def _parse_word(self) -> str:
        self._ws()
        start = self.i
        if start >= len(self.text):
            raise ParseError(self.line_no, self.i + 1, "expected word")
        ch = self.text[self.i]
        if not ch.isalpha():  # must start with a letter
            raise ParseError(self.line_no, self.i + 1, "expected word (letter)")
        self.i += 1
        while self.i < len(self.text):
            ch = self.text[self.i]
            if ch.isalnum() or ch in "_-":
                self.i += 1
            else:
                break
        return self.text[start:self.i]

    # ActorName: read full phrase until ')', then validate against ALLOWED_ACTORS
    def _parse_actor_name(self) -> str:
        self._ws()
        start = self.i
        j = self.i
        while j < len(self.text) and self.text[j] != ')':
            j += 1
        if j >= len(self.text):
            raise ParseError(self.line_no, self.i + 1, "unterminated actor; missing ')'")

        raw = self.text[start:j].strip()

        def _norm(s: str) -> str:
            return " ".join(s.strip().split()).lower()

        allowed_map = {_norm(a): a for a in ALLOWED_ACTORS}
        key = _norm(raw)
        if key not in allowed_map:
            allowed = ", ".join(sorted(ALLOWED_ACTORS, key=str))
            raise ParseError(self.line_no, self.i + 1, f"unknown actor; allowed: {allowed}")

        # Move cursor to just before ')'; caller will consume ')'
        self.i = j
        return allowed_map[key]

    # Comments helpers
    def _inline_comment_opt(self):
        saved = self.i
        self._ws()
        if self._match("#"):
            # consume to end of line
            while not self._eol():
                self.i += 1
        else:
            self.i = saved

    def _step_comment_opt(self):
        saved = self.i
        self._ws()
        if self._match("#"):
            while not self._eol():
                self.i += 1
        else:
            self.i = saved

    def _trailing_opt(self):
        # Trailing := (WS '#' ... )?
        self._step_comment_opt()

    # Arrow
    def _arrow_opt(self) -> bool:
        saved = self.i
        if self._match("→"):
            return True
        if self._match("->"):
            return True
        self.i = saved
        return False

    # Low-level matching utilities
    def _peek(self, s: str) -> bool:
        return self.text.startswith(s, self.i)

    def _peek_char(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def _match(self, s: str) -> bool:
        if self.text.startswith(s, self.i):
            self.i += len(s)
            return True
        return False

    def _expect(self, s: str):
        if not self._match(s):
            col = self.i + 1
            raise ParseError(self.line_no, col, f"expected {s!r}")

    def _peek_word_ci(self, w: str) -> bool:
        return self.text[self.i:].lower().startswith(w.lower())

    def _expect_word_ci(self, w: str):
        n = len(w)
        if not self._peek_word_ci(w):
            col = self.i + 1
            raise ParseError(self.line_no, col, f"expected {w!r}")
        self.i += n

    def _ws(self):
        while self.i < len(self.text) and self.text[self.i] in " \t":
            self.i += 1

    def _ws1(self):
        # consume at least one WS if present; else raise to stop loops
        if self.i < len(self.text) and self.text[self.i] in " \t":
            self.i += 1
            self._ws()
        else:
            raise ParseError(self.line_no, self.i + 1, "expected space")

    def _eol(self) -> bool:
        return self.i >= len(self.text) or self.text[self.i] in "\r\n"
