SYSTEM_PROMPT = """You construct minimal counterfactuals for HotpotQA yes/no comparison questions.

Use ONLY the provided question and supporting facts. Do not use outside knowledge.

First determine whether the example is a valid:
1. same-attribute comparison, or
2. shared-property comparison.

If valid, identify the answer-bearing attribute/property and its value for each
entity, using only the supporting facts.

Then change exactly ONE answer-bearing value/property for exactly ONE entity so
that the answer flips. The question and both entity names must remain unchanged.

For YES -> NO: break the existing equality/shared property by replacing the
intervened entity's value with a different plausible value.

For NO -> YES: create the equality/shared property by replacing the intervened
entity's value with the OTHER entity's original value, whenever possible. Do
not invent a third value here.

The counterfactual fact is a synthetic intervention for testing reasoning, not
a claim about the real world. Preserve as much of the original sentence as
possible; only replace the answer-bearing span.

value_span_1/value_span_2 MUST be copied verbatim from each entity's supporting fact -
never invented or paraphrased. value_1/value_2 are their normalized form, usually
identical but sometimes with extra wrapping (e.g. span "a musician" vs value "musician").
changed_span must equal the intervention_entity's value_span, since that is what gets
replaced. counterfactual_value is the new synthetic replacement and need not be grounded.

Return ONLY one valid JSON object, no markdown fences, no extra text."""

SAME_ATTR_EXAMPLE_Y2N = """Example (same-attribute, YES -> NO, inventing a new value):
Question: Are Mount Tamalpais and Mount Diablo in the same state?
Entity: Mount Tamalpais
Supporting fact: Mount Tamalpais is located in Marin County, California.
Entity: Mount Diablo
Supporting fact: Mount Diablo is located in Contra Costa County, California.
Original answer: yes
Required counterfactual answer: no

Expected output:
{"valid": true, "comparison_type": "same_attribute", "entity_1": "Mount Tamalpais", "entity_2": "Mount Diablo", "attribute": "state", "property": null, "value_1": "California", "value_2": "California", "value_span_1": "California", "value_span_2": "California", "intervention_entity": "Mount Diablo", "original_fact": "Mount Diablo is located in Contra Costa County, California.", "counterfactual_value": "Nevada", "counterfactual_fact": "Mount Diablo is located in Contra Costa County, Nevada.", "changed_span": "California", "original_answer": "yes", "counterfactual_answer": "no", "intervention_type": "break_equality", "reason": "value_span_2 is copied verbatim from the real fact and here equals value_2. 'Nevada' is a synthetic replacement, not grounded in any fact, which is expected for the intervention itself. Replacing California with Nevada makes the two states different, flipping the answer to NO."}"""

SAME_ATTR_EXAMPLE_N2Y = """Example (same-attribute, NO -> YES, reusing the other entity's value):
Question: Are Mount Tamalpais and Mount Diablo in the same state?
Entity: Mount Tamalpais
Supporting fact: Mount Tamalpais is located in Marin County, California.
Entity: Mount Diablo
Supporting fact: Mount Diablo is located in Contra Costa County, Nevada.
Original answer: no
Required counterfactual answer: yes

Expected output:
{"valid": true, "comparison_type": "same_attribute", "entity_1": "Mount Tamalpais", "entity_2": "Mount Diablo", "attribute": "state", "property": null, "value_1": "California", "value_2": "Nevada", "value_span_1": "California", "value_span_2": "Nevada", "intervention_entity": "Mount Diablo", "original_fact": "Mount Diablo is located in Contra Costa County, Nevada.", "counterfactual_value": "California", "counterfactual_fact": "Mount Diablo is located in Contra Costa County, California.", "changed_span": "Nevada", "original_answer": "no", "counterfactual_answer": "yes", "intervention_type": "create_equality", "reason": "value_span_2 is copied verbatim from the real fact and here equals value_2. Replacing it with Mount Tamalpais's own grounded value (California) makes the two states equal, flipping the answer to YES."}"""

SHARED_PROP_EXAMPLE_Y2N = """Example (shared-property, YES -> NO, inventing a new value):
Question: Are A and B both musicians?
Entity: A
Supporting fact: A works as a musician and songwriter.
Entity: B
Supporting fact: B works as a musician known for jazz piano.
Original answer: yes
Required counterfactual answer: no

Expected output:
{"valid": true, "comparison_type": "shared_property", "entity_1": "A", "entity_2": "B", "attribute": null, "property": "musician", "value_1": "musician", "value_2": "musician", "value_span_1": "a musician", "value_span_2": "a musician", "intervention_entity": "B", "original_fact": "B works as a musician known for jazz piano.", "counterfactual_value": "an actor", "counterfactual_fact": "B works as an actor known for jazz piano.", "changed_span": "a musician", "original_answer": "yes", "counterfactual_answer": "no", "intervention_type": "remove_property", "reason": "value_span_2 ('a musician') is the literal phrase including its article, while value_2 ('musician') is the normalized property. Replacing the literal span with 'an actor' removes the shared property, flipping the answer to NO."}"""

SHARED_PROP_EXAMPLE_N2Y = """Example (shared-property, NO -> YES, reusing the other entity's value):
Question: Are A and B both musicians?
Entity: A
Supporting fact: A works as a musician and songwriter.
Entity: B
Supporting fact: B works as an actor known for jazz piano.
Original answer: no
Required counterfactual answer: yes

Expected output:
{"valid": true, "comparison_type": "shared_property", "entity_1": "A", "entity_2": "B", "attribute": null, "property": "musician", "value_1": "musician", "value_2": "actor", "value_span_1": "a musician", "value_span_2": "an actor", "intervention_entity": "B", "original_fact": "B works as an actor known for jazz piano.", "counterfactual_value": "a musician", "counterfactual_fact": "B works as a musician known for jazz piano.", "changed_span": "an actor", "original_answer": "no", "counterfactual_answer": "yes", "intervention_type": "add_property", "reason": "value_span_2 ('an actor') is the literal phrase in the fact. Replacing it with A's phrasing of the shared property ('a musician') creates the shared property, flipping the answer to YES."}"""


def counterfactual_messages(candidate):
    target = "no" if candidate["answer"] == "yes" else "yes"
    is_same_attr = candidate["lexical_type"] == "same_attribute_candidate"
    if is_same_attr:
        example = SAME_ATTR_EXAMPLE_Y2N if target == "no" else SAME_ATTR_EXAMPLE_N2Y
    else:
        example = SHARED_PROP_EXAMPLE_Y2N if target == "no" else SHARED_PROP_EXAMPLE_N2Y

    entities_block = "\n".join(
        f"Entity: {e['entity']}\nSupporting fact: {e['fact']}"
        for e in candidate["entities"]
    )

    user = f"""{example}

Now do the same for this case:

Question: {candidate['question']}
Lexical candidate type: {candidate['lexical_type']}
{entities_block}
Original answer: {candidate['answer']}
Required counterfactual answer: {target}

Return exactly this JSON schema:
{{"valid": true, "comparison_type": "same_attribute" or "shared_property", "entity_1": "...", "entity_2": "...", "attribute": "..." or null, "property": "..." or null, "value_1": "...", "value_2": "...", "value_span_1": "...", "value_span_2": "...", "intervention_entity": "...", "original_fact": "...", "counterfactual_value": "...", "counterfactual_fact": "...", "changed_span": "...", "original_answer": "{candidate['answer']}", "counterfactual_answer": "{target}", "intervention_type": "break_equality" or "create_equality" or "remove_property" or "add_property", "reason": "..."}}

Return {{"valid": false, "reason": "..."}} if:
- this is not actually a same-attribute/shared-property comparison
- the answer-bearing attribute/property cannot be identified
- the relevant value/property is not explicitly supported by the facts
- changing one fact cannot cleanly flip the answer
- the intervention would require changing multiple facts, the question, or the entity names

Do not use outside knowledge."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
