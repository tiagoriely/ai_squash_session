# Sequence DSL — EBNF Spec (human-readable)

This document mirrors `sequence.peg` in a reviewer-friendly EBNF.  
The DSL captures **linear shot sequences** with local options/repeats and a restart marker.

- Parse **one line at a time** (store `rules.sequence` as a string or a list of strings in YAML).
- PEG is deterministic (prioritized choice). This EBNF describes the same language, not a different one.

---

## Lexical / Tokens

- **Arrow**: either the Unicode arrow `→` or ASCII `->`.
- **Names**: bare words (`[A-Za-z][A-Za-z0-9_-]*`) possibly spaced, or `"quoted names"`.
- **Actors** (closed set): `A`, `B`, `initiating player`, `whoever chooses`, `opponent of cross`.
- **Comments**:
  - After an **Action**: `# …` to end of line (inline comment).
  - After **any Step** (before next arrow / end): a **step-level** comment is allowed.
  - Avoid comments **inside parentheses** between alternatives; place them after the group.

- **Whitespace**: spaces/tabs (`WS`) are insignificant between tokens; newlines end a sequence line.

---

## Grammar (EBNF)

```ebnf
Sequence    = WS , Step , { StepComment? , Arrow , Step } , StepComment? , Trailing? , WS ;

Step        = Restart | Optional | Repeat | Choice | Action ;

(* comment after any Step, before next arrow or end *)
StepComment = [ WS ] , "#" , { not-Newline } ;

(* separators / control *)
Arrow       = WS , ( "→" | "->" ) , WS ;

Restart     = "restart" , WS , [ FreeText ] ;

(* allow optional of a single Action OR a parenthesized group *)
Optional    = "optional:" , WS , ( Action | Group ) ;

(* grouping constructs *)
Repeat      = Group , WS , "*" ;
Group       = "(" , WS , ( ChoiceInner | Action ) , WS , ")" ;
Choice      = "(" , WS , ChoiceInner , WS , ")" ;
ChoiceInner = Action , { WS , "|" , WS , Action } ;

(* atoms *)
Action      = Name , [ Actor ] , [ InlineComment ] ;
Actor       = WS , "(" , WS , ActorName , WS , ")" ;
ActorName   = "A" | "B" | "initiating player" | "whoever chooses" | "opponent of cross" ;

Name        = Quoted | Word , { WS , Word } ;
Quoted      = '"' , { '\\"' | not-quote-not-newline } , '"' ;
Word        = letter , { letter | digit | "_" | "-" } ;

(* comments & whitespace *)
InlineComment = [ WS ] , "#" , { not-Newline } ;
FreeText      = { not-Newline }+ ;
WS            = { " " | "\t" } ;
Trailing      = ( WS , "#" , { not-Newline } ) ;
