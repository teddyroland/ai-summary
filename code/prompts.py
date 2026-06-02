"""Prompt templates for the three pipeline stages.

Each stage uses two prompts:
- The "list" prompt asks the model to produce a list of conditions (questions
  for passage selection, requirements for summaries).
- The "item" prompt takes one condition at a time and asks the model to apply
  it (select a passage, write a summary).

The scene-setting and global-theorizing summary stages share the same item
prompt template, so we have six total templates (not seven).

Each template uses [[ PLACEHOLDER ]] markers exactly as written in the project
specification. The render functions below fill those in.
"""

# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

# A short instruction prepended to every call so the model returns JSON in the
# expected shape. Stage-specific schemas are enforced more rigorously by
# OpenAI's Structured Outputs; for Bedrock models we rely on this text plus
# json.loads with a self-correcting retry.
SYSTEM_PROMPT = (
    "You are an assistant for literary studies research. "
    "Respond with valid JSON matching the requested schema. "
    "Do not include any text outside the JSON object."
)


# ---------------------------------------------------------------------------
# Stage 4a — Passage selection
# ---------------------------------------------------------------------------

QUESTIONS_PROMPT = """Please generate a list of high-level interpretive questions for the text below.

Author: [[ AUTHOR ]]

Title: [[ TITLE ]]

Complete Text: [[ TEXT ]]

Here are some examples of high-level interpretive questions posed to different texts:

1. What distinct categories of events occur in the text?
2. How did the author feel while writing the text?
3. How do the words sound when read aloud?
4. What is the good life?
5. What is marriage as an institution?
6. How do emotions become literary forms?
7. What is modernism's relationship to post-colonialism?

Please generate a list of [[ SELECTION_NUMBER ]] high-level interpretive questions for the text above, in JSON format.

Questions: """


PASSAGE_PROMPT = """Select a single poem or a 100-200 word passage from the following text, which addresses the interpretive question.

Author: [[ AUTHOR ]]

Title: [[ TITLE ]]

Question: [[ QUESTION ]]

Complete Text: [[ TEXT ]]

Select a single poem or a 50-200 word passage from the following text, that best addresses the interpretive question. Return ONLY a verbatim excerpt — a literal, consecutive copy of words taken directly from the source. Do not add commentary, analysis, framing, headings, or explanation; do not paraphrase, summarize, or modernize the text.

Poem/Passage: """


# A follow-up instruction appended to PASSAGE_PROMPT when the model's first
# response failed the verbatim-substring check. Used by pipeline.py.
PASSAGE_VERBATIM_RETRY_INSTRUCTION = (
    "\n\nYour previous response included text that does not appear verbatim "
    "in the source. Return ONLY a literal, consecutive excerpt copied directly "
    "from the source above. No commentary, analysis, or framing."
)


# ---------------------------------------------------------------------------
# Stage 4b — Scene-setting summaries
# ---------------------------------------------------------------------------

SCENE_REQUIREMENTS_PROMPT = """Please generate a list of specific summary requirements for the passage below.

Author: [[ AUTHOR ]]

Title (Complete Text): [[ TITLE ]]

Passage: [[ PASSAGE ]]

Here are some requirement examples based on different passages:

1. Summarize the linguistic register of the passage.
2. Summarize the identities of characters in the passage and their relationships.
3. Summarize the locations where events in the passage take place.
4. Summarize situations or events precipitate the events in the passage.
5. Provide a psychoanalytic summary of the passage.
6. Summarize the author's biographical context for the passage.
7. Summarize the metrical form of a poem.
8. Summarize the grammatical syntax.
9. Summarize the sounds that occur in the passage.

Please generate a list of [[ SCENE_NUMBER ]] specific summary requirements for the text above, in JSON format.

Requirements: """


# ---------------------------------------------------------------------------
# Stage 4c — Global-theorizing summaries
# ---------------------------------------------------------------------------

GLOBAL_REQUIREMENTS_PROMPT = """Please generate a list of specific summary requirements for the passage below.

Author: [[ AUTHOR ]]

Title (Complete Text): [[ TITLE ]]

Passage: [[ PASSAGE ]]

Here are some requirement examples based on different passages:

1. Summarize how art changed in the modern era.
2. Summarize the worldview of pastoral poetry.
3. Summarize how heterosexual courtship be slowed down, thwarted, and avoided in Jane Austen's novels.
4. Summarize how aestheticism work against heterosexuality in Jane Austen's novels.
5. Summarize how artworks, from roughly 1800 to the present, represent anxiety as a masculine emotion.
6. Summarize how artworks represent emotions by assimilating the theories of nineteenth- and twentieth-century philosophers, psychoanalysts, and cultural critics.
7. Summarize how postcolonial modernist artworks practice diffusionism, spreading modernist innovations to the periphery.
8. Summarize how postcolonial modernist artworks practice resistance, struggling against modernism as a colonial imposition.

Please generate a list of [[ GLOBAL_NUMBER ]] specific summary requirements for the text above, in JSON format.

Requirements: """


# ---------------------------------------------------------------------------
# Shared item prompt for both scene-setting and global-theorizing summaries
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """Summarize the following passage based on the specific requirement. If necessary, use the complete text from which the passage is drawn for additional context.

Author: [[ AUTHOR ]]

Title (Complete Text): [[ TITLE ]]

Passage: [[ PASSAGE ]]

Requirement: [[ REQUIREMENT ]]

Complete Text: [[ TEXT ]]

Summarize the passage above based on the specific requirement. If necessary, use the complete text from which the passage is drawn for additional context. Output the summary into JSON format.

Summary: """


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------
#
# Each render function takes the relevant context (metadata, text, condition)
# and returns a finished user-prompt string. We use plain str.replace rather
# than str.format or jinja so the [[ PLACEHOLDER ]] markers in the spec stay
# intact in the source code and are easy to grep for.


def _fill(template: str, replacements: dict[str, str]) -> str:
    """Replace each [[ KEY ]] marker with its value."""
    result = template
    for key, value in replacements.items():
        result = result.replace(f"[[ {key} ]]", str(value))
    return result


def render_questions_prompt(author: str, title: str, text: str, selection_n: int) -> str:
    return _fill(
        QUESTIONS_PROMPT,
        {
            "AUTHOR": author,
            "TITLE": title,
            "TEXT": text,
            "SELECTION_NUMBER": selection_n,
        },
    )


def render_passage_prompt(author: str, title: str, question: str, text: str) -> str:
    return _fill(
        PASSAGE_PROMPT,
        {
            "AUTHOR": author,
            "TITLE": title,
            "QUESTION": question,
            "TEXT": text,
        },
    )


def render_scene_requirements_prompt(author: str, title: str, passage: str, scene_n: int) -> str:
    return _fill(
        SCENE_REQUIREMENTS_PROMPT,
        {
            "AUTHOR": author,
            "TITLE": title,
            "PASSAGE": passage,
            "SCENE_NUMBER": scene_n,
        },
    )


def render_global_requirements_prompt(author: str, title: str, passage: str, global_n: int) -> str:
    return _fill(
        GLOBAL_REQUIREMENTS_PROMPT,
        {
            "AUTHOR": author,
            "TITLE": title,
            "PASSAGE": passage,
            "GLOBAL_NUMBER": global_n,
        },
    )


def render_summary_prompt(
    author: str, title: str, passage: str, requirement: str, text: str
) -> str:
    """Shared by both scene-setting and global-theorizing summary stages."""
    return _fill(
        SUMMARY_PROMPT,
        {
            "AUTHOR": author,
            "TITLE": title,
            "PASSAGE": passage,
            "REQUIREMENT": requirement,
            "TEXT": text,
        },
    )
