# rule: salience@v2
# NOT an LLM prompt. A deterministic, purely lexical scoring formula.
# Registered and hashed like prompts. The policy references it via
# `triage.formula`; policy canonicalization resolves that reference to
# this file's content hash, so `policy_hash` pins the exact formula in
# force (SPEC §8). Forking this file is a policy change, visible in every
# subsequent event.
#
# Division of labor: this file defines the SIGNALS and the FUNCTIONAL
# FORM. Weights, threshold, caps and role exclusions are TUNING and live
# in memory.policy.yaml (covered by policy_hash independently).
#
# v2 keeps the same lexical-only guarantees as v1, but adds explicit
# short-turn damping because the stopword-ratio proxy saturates on terse
# replies that DMF's POS-based density would not treat as equally salient.

## Determinism requirements (normative)

- NO NLP models, NO embeddings, NO external resources. The complete input
  surface is: Unicode NFC normalization, the tokenizer below, the stopword
  list embedded in this file, and the regex cue patterns below.
  Identical text MUST produce an identical score on any machine —
  `memledger rebuild` depends on this.
- DMF's POS-based information density and NER entity count are
  deliberately NOT used in v2: tagger output varies across model versions
  and would break rebuild reproducibility. A future `salience@v3` may use
  them only by pinning the NLP model and recording its digest, exactly as
  LLM calls record `model_digest`.
- DMF's semantic-divergence penalty is deliberately omitted: it requires
  embeddings, whose model dependence belongs to the disposable index
  layer, not to a formula folded into `policy_hash`.

## Tokenization (normative)

1. Normalize the turn text to Unicode NFC.
2. Tokens are maximal matches of the regex: `[^\W_][^\W_’'-]*`
3. Sentence-initial positions: token index 0, and any token immediately
   following a `.`, `!` or `?` character.
4. Case: stopword matching is on the lowercased token; the entity proxy
   inspects original casing.

## Signals

### 1. lexical_density ∈ [0, 1]
Proxy for DMF's information density:
    lexical_density = 1 − (stopword_tokens / total_tokens)
0 if the turn has no tokens. A token is a stopword iff its lowercase form
is in the embedded list `stopwords.en@v1` below.

### 2. token_count ∈ N and length_norm ∈ [0, 1]
Short-turn calibration terms:
    token_count  = total_tokens
    length_norm  = min(token_count, density_length_cap) / density_length_cap
`density_length_cap` defaults to 6. If the turn has no tokens,
`length_norm = 0`.

### 3. entity_norm ∈ [0, 1]
Entity proxy count E = number of tokens that are any of:
  (a) uppercase-initial AND not sentence-initial;
  (b) containing at least one digit;
  (c) ALL-CAPS of length 2–6 (acronyms: "API", "GDPR").
    entity_norm = min(E, entity_cap) / entity_cap        # default cap: 5

### 4. cues — pragmatic classes (English adapter)
Case-insensitive regex per class. `cue_hits` = number of DISTINCT classes
matched; `cues_norm = min(cue_hits, cue_cap) / cue_cap` (default cap: 2).

| class         | pattern |
|---------------|---------|
| preference    | `\b(i\|we)\s+(really\s+)?(prefer\|like\|love\|hate\|dislike)\b` or `\bmy\s+favou?rite\b` |
| constraint    | `\b(never\|always\|must(\s+not)?\|do\s+not\|don'?t\|avoid\|only\s+use\|make\s+sure)\b` |
| correction    | `\bactually\b` or `\bi\s+meant\b` or `\bthat'?s\s+(not\s+right\|wrong)\b` or `\bcorrection\b` or `^no[,.]` |
| replacement   | `\bnot\s+\w+\s+but\b` or `\binstead\s+of\b` or `\bswitch(ed)?\s+to\b` or `\bfrom\s+now\s+on\b` |
| current_state | `\bcurrently\b` or `\bright\s+now\b` or `\bas\s+of\s+(now\|today)\b` or `\bthese\s+days\b` |

Matched classes are recorded in the `triaged` event and drive the
`always_extract_cues` bypass. Non-English text scores on density and
entities alone (cue_hits = 0) until a language adapter is forked; forks
are new registry entries (e.g. `salience@v2-it`) and therefore policy
changes.

## Formula

    density_signal = lexical_density · length_norm^density_length_power
    z              = w_density  · density_signal
                   + w_entities · entity_norm
                   + w_cues     · cues_norm
    salience       = σ(z − x0) = 1 / (1 + e^−(z − x0))

Policy defaults: density_length_cap = 6, density_length_power = 2.0,
w_density = 3.0, w_entities = 2.0, w_cues = 1.5, x0 = 1.5,
threshold = 0.35. Calibration anchors with those defaults: an all-stopword
phatic turn scores σ(−1.5) ≈ 0.18; a one-token dense filler like "noted"
scores ≈ 0.19; "sounds good to me" lands ≈ 0.30; "the deploy failed
because of the env vars" lands ≈ 0.50.

## Verdict (normative evaluation order)

1. `role ∈ triage.ineligible_roles`                  → **ineligible**
2. any matched class ∈ `triage.always_extract_cues`  → **extract** (bypass)
3. `salience ≥ triage.threshold`                     → **extract**
4. otherwise                                         → **skip**

Semantics: **skip** = "below threshold this time" — recoverable by
`memledger regenerate` after lowering the threshold. **ineligible** =
"this class of turns is never extracted by policy" — NOT recovered by a
threshold change; it re-enters only if `ineligible_roles` changes.

## Recorded signals (payload of the `triaged` event)

    {lexical_density, token_count, length_norm, entity_norm, cue_classes: [...], salience}

## stopwords.en@v1 (part of this file's hash)

a about above after again all am an and any are as at be because been
before being below between both but by can could did do does doing down
during each few for from further had has have having he her here hers
him his how i if in into is it its just me more most my no nor not of
off on once only or other our ours out over own same she should so some
such than that the their theirs them then there these they this those
through to too under until up very was we were what when where which
while who whom why will with would you your yours
yeah yes ok okay hi hello hey thanks thank please sure got right cool
great awesome nice fine well hmm oh ah wow bye goodbye