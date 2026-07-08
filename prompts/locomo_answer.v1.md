# prompt: locomo_answer@v1
# schema: locomo_answer@v1
# params: temperature=0, response constrained to JSON
# input placeholders: {{speakers}}, {{question_hint}}, {{question}},
#                     {{memory_context}}

You are answering a benchmark question about a long-running conversation.
Use only the memory statements and evidence below.

Rules:

1. If the answer is supported, return a short answer phrase using the same
   wording as the memory or evidence whenever possible.
2. If the answer is not supported by the retrieved memory, answer exactly
   "No information available".
3. Do not explain your reasoning.
4. Do not invent dates, names or details that are not grounded in the memory
   context.

Return ONLY a JSON object:

{
  "answer": "short answer here"
}

Conversation speakers: {{speakers}}

Question-specific guidance:
{{question_hint}}

Question:
{{question}}

Retrieved memory:
{{memory_context}}