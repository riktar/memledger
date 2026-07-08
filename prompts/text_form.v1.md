# templates: text_form@v1
# NOT an LLM prompt. Deterministic string templates, applied by the
# framework to generate the indexable sentence of each tuple.
# Registered and hashed like prompts so text_form is reproducible.
#
# Resolution order: exact relation match -> category match -> fallback.
# Placeholders: {subject}, {relation}, {value}, {when}, {context}.
# {relation_words} = relation with underscores replaced by spaces.
# Qualifier suffixes are appended when present:
#   when    -> " (as of {when})"
#   context -> " in the context of {context}"

[exact]
preferred_language   = "{subject} prefers {value} as their language"
works_at             = "{subject} works at {value}"
role                 = "{subject}'s role is {value}"
name                 = "{subject}'s name is {value}"
timezone             = "{subject} is in the {value} timezone"
constraint           = "{subject} has a standing constraint: {value}"
decision             = "{subject} decided: {value}"
outcome_failure      = "{value} failed for {subject}"
outcome_success      = "{value} succeeded for {subject}"
alias                = "{subject} is also known as {value}"
deadline             = "{subject} has a deadline: {value}"

[category]
preferred_*          = "{subject} prefers {value} as {relation_words#preferred_}"
uses_*               = "{subject} uses {value} as {relation_words#uses_}"
dislikes_*           = "{subject} dislikes {value}"

[fallback]
default              = "{subject}: {relation_words} is {value}"

# Extraction note: extract@v1 also emits a text_form; the framework
# OVERWRITES it with these templates when a template resolves, and keeps
# the LLM's sentence only for fallback-matched relations where the
# template output would be awkward. Either way, the chosen generator
# (template id or "llm") is recorded in the tuple's provenance.