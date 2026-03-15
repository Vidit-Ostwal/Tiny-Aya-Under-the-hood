import json
import os
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


class SentencePair(BaseModel):
    sentence_1: str
    sentence_2: str


class SentencePairBatch(BaseModel):
    pairs: List[SentencePair]


LEXICAL_SYSTEM_PROMPT = """You are a computational linguistics expert specializing in controlled dataset construction for NLP probing experiments. Your task is to generate high-quality lexical variation sentence pairs for a linguistic probing study.

TASK
Generate sentence pairs where sentence_1 and sentence_2 are identical in meaning and syntactic structure, differing only in a single content word replaced by a synonym or near-synonym.

RULES
1. The two sentences must be grammatically identical in structure — same word order, same clause structure, same morphology everywhere except the swapped word.
2. The swapped words must be true synonyms or near-synonyms in the given context (not just related words).
3. Sentences should be everyday, concrete, and natural - no jargon, no abstract philosophy.
4. Vary the swapped word class across pairs: target nouns, verbs, adjectives, and adverbs roughly equally.
5. Sentences should be 8–18 words long.
6. No pair should repeat a word swap used in another pair.
7. Do not generate pairs where the swap changes register significantly (e.g. formal vs informal).

EXAMPLES
{"type":"lexical","sentence_1":"She purchased a new jacket last week.","sentence_2":"She bought a new jacket last week."}
{"type":"lexical","sentence_1":"The physician examined the patient carefully.","sentence_2":"The doctor examined the patient carefully."}
{"type":"lexical","sentence_1":"He replied to the message immediately.","sentence_2":"He responded to the message right away."}
{"type":"lexical","sentence_1":"She felt fatigued after the long hike.","sentence_2":"She felt tired after the long hike."}"""

SYNTACTIC_SYSTEM_PROMPT = """You are a computational linguistics expert specializing in controlled dataset construction for NLP probing experiments. Your task is to generate high-quality syntactic variation sentence pairs for a linguistic probing study.

TASK
Generate sentence pairs where sentence_1 and sentence_2 are grammatically transformed equivalents. They must describe the same event or state of affairs. The transformation must be a recognized grammatical alternation (see phenomena list below).

RULES
1. The relationship must be truth-conditionally equivalent — they describe the same event or state of affairs.
2. Sentences should differ meaningfully in structure, not just word order.
3. Sentences should be everyday, concrete, and natural — no jargon, no abstract philosophy.
4. Vary the phenomena (listed below) across pairs — don't pile up one type.
5. Sentences should be 8–18 words long.
6. No pair should reproduce more than 30% of the wording of another pair.

PHENOMENA TO COVER (cycle through these):
- Active / passive: "She baked the cake." / "The cake was baked by her."
- Cleft: "She found the key." / "It was she who found the key."
- Subject raising: "It seems that she won." / "She seems to have won."
- Extraposition: "That she resigned surprised us." / "It surprised us that she resigned."
- Attributive / predicative: "The visible star..." / "The star is visible..."
- Dative alternation: "She gave the book to him." / "She gave him the book."
- Tough movement: "It is hard to please her." / "She is hard to please."
- Temporal clause reordering: "After eating, she left." / "She left after eating."
- Genitive alternation: "The book of the student." / "The student's book."

EXAMPLES
{"type":"syntactic","sentence_1":"The manager approved the new plan yesterday.","sentence_2":"The new plan was approved by the manager yesterday."}
{"type":"syntactic","sentence_1":"She gave the report to her colleague.","sentence_2":"She gave her colleague the report."}
{"type":"syntactic","sentence_1":"After finishing her coffee, she left the office.","sentence_2":"She left the office after finishing her coffee."}
{"type":"syntactic","sentence_1":"It surprised everyone that he refused the offer.","sentence_2":"That he refused the offer surprised everyone."}"""

SEMANTIC_SYSTEM_PROMPT = """You are a computational linguistics expert specializing in controlled dataset construction for NLP probing experiments. Your task is to generate high-quality semantic variation sentence pairs for a linguistic probing study.

TASK
Generate sentence pairs where sentence_1 and sentence_2 are semantically related. They must describe the same event or state, but differ substantially in both vocabulary and syntactic form.

RULES
1. The sentences must differ meaningfully in both vocabulary and structure — not just a synonym swap (that's lexical) or a passive/active transformation (that's syntactic).
2. The relationship must be clear and unambiguous — avoid pairs where the paraphrase relationship could be debatable or context-dependent.
3. Sentences should be everyday, concrete, and natural — no jargon, no abstract philosophy.
4. Sentences should be 8–18 words long.
5. Be conservative: if the equivalence could be disputed, discard the pair.

PHENOMENA TO COVER (cycle through these):
- Negation paraphrase: "The meeting was canceled." / "The meeting did not take place."
- Entailment: "The car stopped." / "The car is no longer moving."
- Scalar implicature: "He ate some of the cake." / "He did not eat all of the cake."
- Existential paraphrase: "There are no seats left." / "All seats are taken."
- Resultative: "She locked the door." / "The door is now locked."
- Aspectual paraphrase: "He has finished the report." / "The report is done."

EXAMPLES
{"type":"semantic","sentence_1":"The meeting was canceled.","sentence_2":"The meeting did not take place."}
{"type":"semantic","sentence_1":"All of the guests arrived on time.","sentence_2":"None of the guests were late."}
{"type":"semantic","sentence_1":"She has finished all her work.","sentence_2":"No work remains for her to do."}
{"type":"semantic","sentence_1":"The store ran out of milk.","sentence_2":"There was no milk left at the store."}"""

VARIATION_SYSTEM_PROMPTS = {
    "lexical": LEXICAL_SYSTEM_PROMPT,
    "syntactic": SYNTACTIC_SYSTEM_PROMPT,
    "semantic": SEMANTIC_SYSTEM_PROMPT,
}

VARIATION_TYPES = ["lexical", "syntactic", "semantic"]


class LinguisticVariationPipeline:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1",
        batch_size: int = 25,
    ):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.batch_size = batch_size

    def generate_batch(self, variation_type: str, batch_size: int, recent_pairs: List[dict]) -> List[SentencePair]:
        # Inject the last 4 generated pairs so the model avoids repeating them
        recency_note = ""
        if recent_pairs:
            examples = "\n".join(
                f'  "{p["sentence_1"]}" / "{p["sentence_2"]}"'
                for p in recent_pairs[-4:]
            )
            recency_note = f"\n\nAvoid generating pairs similar to these recently generated ones:\n{examples}"

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": VARIATION_SYSTEM_PROMPTS[variation_type]},
                {"role": "user", "content": (
                    f"Generate exactly {batch_size} {variation_type} variation sentence pairs. "
                    f"Ensure variety in topics, sentence length, and phenomena types across the batch."
                    + recency_note
                )},
            ],
            response_format=SentencePairBatch,
        )
        return completion.choices[0].message.parsed.pairs

    def generate(
        self,
        pairs_per_type: int = 1000,
        output_path: Optional[str | Path] = None,
        interactive: bool = True,
    ) -> List[dict]:
        all_results = []
        pair_counter = 1

        for variation_type in VARIATION_TYPES:
            print(f"\n--- {variation_type.upper()} ({pairs_per_type} pairs) ---")
            generated = []
            seen_sentences = set()
            stopped_early = False

            while len(generated) < pairs_per_type:
                remaining = pairs_per_type - len(generated)
                batch_size = min(self.batch_size, remaining)

                raw_batch = self.generate_batch(variation_type, batch_size, generated)

                # Dedup: drop any pair whose sentence_1 we've seen before
                new_pairs = []
                for pair in raw_batch:
                    key = pair.sentence_1.strip().lower()
                    if key not in seen_sentences:
                        seen_sentences.add(key)
                        new_pairs.append(pair)

                generated.extend(new_pairs)
                duplicates_dropped = len(raw_batch) - len(new_pairs)

                if interactive:
                    print(f"\nBatch ({len(new_pairs)} new, {duplicates_dropped} duplicates dropped):")
                    for i, pair in enumerate(new_pairs, 1):
                        print(f"  {i}. [{variation_type}]")
                        print(f"     s1: {pair.sentence_1}")
                        print(f"     s2: {pair.sentence_2}")
                    print(f"\n  Progress: {len(generated)}/{pairs_per_type}")
                    choice = input("  [Enter] continue  |  [q] stop this type  |  [Q] stop everything: ").strip().lower()
                    if choice == "q":
                        print(f"  Stopping {variation_type} early with {len(generated)} pairs.")
                        stopped_early = True
                        break
                    elif choice == "qq" or choice == "Q":
                        print("  Stopping generation entirely.")
                        for pair in generated:
                            all_results.append({"pair_id": str(pair_counter), "type": variation_type, "sentence_1": pair.sentence_1, "sentence_2": pair.sentence_2})
                            pair_counter += 1
                        self._save(all_results, output_path)
                        return all_results
                else:
                    print(f"  {variation_type}: {len(generated)}/{pairs_per_type} (+{len(new_pairs)}, -{duplicates_dropped} dupes)")

            for pair in generated[:pairs_per_type]:
                all_results.append({
                    "pair_id": str(pair_counter),
                    "type": variation_type,
                    "sentence_1": pair.sentence_1,
                    "sentence_2": pair.sentence_2,
                })
                pair_counter += 1

        self._save(all_results, output_path)
        return all_results

    def _save(self, results: List[dict], output_path: Optional[str | Path]):
        if not output_path:
            return
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} pairs to {path}")


if __name__ == "__main__":
    pipeline = LinguisticVariationPipeline(model="gpt-4.1", batch_size=25)

    results = pipeline.generate(
        pairs_per_type=1000,
        output_path="uth/data/linguistic_variation.json",
        interactive=True,
    )
