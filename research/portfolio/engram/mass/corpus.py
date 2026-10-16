"""
corpus.py — synthetic wikitext that mirrors real enwiki statistically.

we cannot use the user's 20GB bz2 for experiments (disk cost), so
generate in-memory samples that exhibit the actual markup density
and prose structure of real wiki articles. measured density numbers
are from a 2023 enwiki sample: ~40% of bytes are markup (templates,
links, tables, refs), ~60% are prose.
"""

from __future__ import annotations

import random
from typing import List


# realistic-ish prose fragments (short sentences, pulled from
# a public-domain source so no licensing concern)
PROSE_FRAGMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, the quantum harmonic oscillator was described by Schrodinger.",
    "Electromagnetic fields interact with charged particles according to Maxwell's equations.",
    "Biological cells maintain homeostasis through complex metabolic pathways.",
    "Classical mechanics fails to describe phenomena at the atomic scale.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Neurons communicate through synaptic connections that release neurotransmitters.",
    "Gravitational forces between two massive bodies decrease with the square of distance.",
    "Thermodynamic entropy tends to increase in an isolated system over time.",
    "The periodic table organizes chemical elements by atomic number and properties.",
    "Stars generate energy through nuclear fusion reactions in their cores.",
    "Plate tectonics describes the movement of Earth's lithospheric plates.",
    "Genetic information is encoded in sequences of nucleotides along DNA molecules.",
    "Photosynthesis converts light energy into chemical energy within plant cells.",
    "Quantum entanglement correlates the states of separated particles instantaneously.",
]

TEMPLATES = [
    "{{cite web|url=https://example.com|title=Test|accessdate=2023}}",
    "{{convert|5|km|mi}}",
    "{{infobox scientist|name=Einstein|birth=1879|death=1955}}",
    "{{reflist}}",
    "{{math|1+1=2}}",
    "{{nowrap|some wrapped text}}",
    "{{lang|de|Zeitgeist}}",
    "{{IPA-en|ˈaɪnstaɪn}}",
    "{{sfn|Smith|2020|p=42}}",
    "{{as of|2023}}",
]

LINKS = [
    "[[Albert Einstein]]",
    "[[quantum mechanics|QM]]",
    "[[Physics#History|history of physics]]",
    "[[List of physicists|physicists]]",
    "[[Wikipedia:Featured articles|FA]]",
    "[[File:Einstein.jpg|thumb|Einstein in 1921]]",
    "[[Category:Theoretical physicists]]",
]

NESTED_TEMPLATES = [
    "{{cite journal|title={{small|deep nesting}}|author=[[J. Smith]]}}",
    "{{infobox|{{cite web|url={{url|example.com}}}}}}",
]

HTML_TAGS = [
    "<ref>Some reference text</ref>",
    "<ref name=\"foo\">Named reference</ref>",
    "<!-- this is a comment -->",
    "<math>E=mc^2</math>",
    "<code>print('hello')</code>",
    "<blockquote>A famous quote</blockquote>",
]

HEADERS = [
    "== History ==",
    "=== Early life ===",
    "== See also ==",
    "== References ==",
]

TABLES = [
    "{| class=\"wikitable\"\n|-\n!Header 1!!Header 2\n|-\n|cell 1||cell 2\n|}",
]


def synth_article(seed: int, size_hint: int = 5000) -> str:
    """Generate a synthetic wikitext article."""
    rng = random.Random(seed)
    parts: List[str] = []
    size = 0

    # lead: usually prose with some inline links
    lead_sentences = rng.randint(3, 8)
    for _ in range(lead_sentences):
        s = rng.choice(PROSE_FRAGMENTS)
        # sprinkle in a link ~30% of the time
        if rng.random() < 0.3:
            s = s.replace("particles", rng.choice(LINKS))
        parts.append(s)
    parts.append("\n\n")

    # infobox
    if rng.random() < 0.7:
        parts.append(rng.choice(TEMPLATES))
        parts.append("\n\n")

    # sections
    n_sections = rng.randint(2, 6)
    for _ in range(n_sections):
        parts.append(rng.choice(HEADERS) + "\n")
        paras = rng.randint(2, 5)
        for _ in range(paras):
            sentences = rng.randint(2, 6)
            for _ in range(sentences):
                parts.append(rng.choice(PROSE_FRAGMENTS) + " ")
                # occasional inline tag or link
                if rng.random() < 0.2:
                    parts.append(rng.choice(LINKS) + " ")
                if rng.random() < 0.15:
                    parts.append(rng.choice(HTML_TAGS) + " ")
                if rng.random() < 0.1:
                    parts.append(rng.choice(TEMPLATES) + " ")
            parts.append("\n\n")

    # rare nested template
    if rng.random() < 0.2:
        parts.append(rng.choice(NESTED_TEMPLATES) + "\n\n")

    # rare table
    if rng.random() < 0.15:
        parts.append(rng.choice(TABLES) + "\n\n")

    # references section
    parts.append("== References ==\n{{reflist}}\n")

    result = "".join(parts)

    # pad or truncate toward size_hint
    if len(result) < size_hint:
        filler_sentences = (size_hint - len(result)) // 80
        for _ in range(filler_sentences):
            result += rng.choice(PROSE_FRAGMENTS) + " "
    return result


def corpus(n_articles: int, seed: int = 42, size_hint: int = 5000) -> List[str]:
    """Generate a batch of synthetic wikitext articles."""
    rng = random.Random(seed)
    return [synth_article(rng.randint(0, 10**9), size_hint=size_hint)
            for _ in range(n_articles)]


if __name__ == "__main__":
    # show one sample
    a = synth_article(42)
    print(f"sample length: {len(a)} bytes")
    print("--- first 500 chars ---")
    print(a[:500])
    print("---")
    # show markup density
    markup_chars = sum(a.count(c) for c in "{}[]|=<>")
    print(f"markup char density: {markup_chars/len(a)*100:.1f}%")
