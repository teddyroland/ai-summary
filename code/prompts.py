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

QUESTIONS_PROMPT = """Please generate a list of high-level interpretive questions for the text below. The questions should identify what the given text can teach us, in terms that are historical, ethical, or aesthetic. The questions should be general enough that they can be posed to other texts as well.

Author: [[ AUTHOR ]]

Title: [[ TITLE ]]

Complete Text: [[ TEXT ]]

Here are some examples of high-level interpretive questions posed to different texts:

1. What is the relationship between our interior lives and exterior events?
2. How did the author feel while writing?
3. How do the words sound when read aloud?
4. What is the good life?
5. What is marriage as an institution?
6. How do emotions become literary forms?
7. What is modernism's relationship to post-colonialism?

Please generate a list of [[ SELECTION_NUMBER ]] high-level interpretive questions for the text above, in JSON format.

Questions: """


PASSAGE_PROMPT = """Select a single poem or a passage of between 100 and 300 words from the following text, which best addresses the requirement.

Author: [[ AUTHOR ]]

Title: [[ TITLE ]]

Requirement: [[ REQUIREMENT ]]

Complete Text: [[ TEXT ]]

Select a single poem or a passage of between 100 and 300 words from the above text, which best addresses the requirement. The length is a hard constraint: count the words and do not exceed 300. Return only a verbatim excerpt: a literal, consecutive copy of words taken directly from the source. Do not add commentary, analysis, framing, headings, or explanation. Do not paraphrase, summarize, or modernize the text.

Poem/Passage: """


# Follow-up instructions appended to PASSAGE_PROMPT when the first response
# fails a check. Used by pipeline.py to drive a stricter re-prompt.
PASSAGE_VERBATIM_RETRY_INSTRUCTION = (
    "\n\nYour previous response included text that does not appear verbatim "
    "in the source. Return ONLY a literal, consecutive excerpt copied directly "
    "from the source above. No commentary, analysis, or framing."
)


def passage_length_retry_instruction(actual_word_count: int) -> str:
    """Length-correction follow-up. Includes the actual word count so the
    model sees how far off it was."""
    return (
        f"\n\nYour previous response was {actual_word_count} words. "
        f"Return a single passage of between 100 and 300 words. "
        f"This is a hard constraint: count the words and do not exceed 300."
    )


# ---------------------------------------------------------------------------
# Stage 4b — Scene-setting summaries
# ---------------------------------------------------------------------------

SCENE_REQUIREMENTS_PROMPT = """Please generate a list of specific summary requirements for the passage below. The requirements should identify the most important features of the passage. The requirements should be general enough that they can be posed to other texts as well.

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

GLOBAL_REQUIREMENTS_PROMPT = """Please generate a list of specific summary requirements for the passage below. The requirements should identify what the passage can teach us, in terms of its author's body of work, its literary genre, or its historical background. The requirements should be general enough that they can be posed to other texts as well.

Author: [[ AUTHOR ]]

Title (Complete Text): [[ TITLE ]]

Passage: [[ PASSAGE ]]

Here are some requirement examples based on different passages:

1. Summarize how the poem claims art changed in the modern era.
2. Summarize the the sonnet expresses worldview of pastoral poetry.
3. Summarize how the passage dramatizes ways in which heterosexual courtship be slowed down, thwarted, and avoided in Jane Austen's novels.
4. Summarize how the passage shows how aestheticism works against heterosexuality in Jane Austen's novels.
5. Summarize how the passage, as an examplary artwork of the past two centuries, represents anxiety as a masculine emotion.
6. Summarize how the film represents emotions by assimilating the theories of nineteenth- and twentieth-century philosophers, psychoanalysts, and cultural critics.
7. Summarize how the poem practices the diffusionism of postcolonial modernism, spreading modernist innovations to the periphery.
8. Summarize how the poem's title practices the resistance of postcolonial modernism, struggling against modernism as a colonial imposition.

Please generate a list of [[ GLOBAL_NUMBER ]] specific summary requirements for the text above, in JSON format.

Requirements: """


# ---------------------------------------------------------------------------
# Shared item prompt for both scene-setting and global-theorizing summaries
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """Summarize the passage below, based on the stated requirement.

Author: [[ AUTHOR ]]

Title (Complete Text): [[ TITLE ]]

Passage: [[ PASSAGE ]]

Requirement: [[ REQUIREMENT ]]

Complete Text: [[ TEXT ]]

Summarize the passage above, based on the stated requirement. If necessary, use the complete text from which the passage is drawn for additional information. Output the summary into JSON format.

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


def render_passage_prompt(author: str, title: str, requirement: str, text: str) -> str:
    return _fill(
        PASSAGE_PROMPT,
        {
            "AUTHOR": author,
            "TITLE": title,
            "REQUIREMENT": requirement,
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
