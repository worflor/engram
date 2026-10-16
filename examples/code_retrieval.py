"""Retrieve actual authored code hunks with byte dynamics + identifier evidence.

No language model. The byte coordinate system is learned from indexed hunks;
queries are not used to train it. Results return retained source evidence.
"""

from engram_retrieval import SignatureIndex, code_tokens
from engram_training import LookupSource, learn_byte_embedding

HUNKS = {
    "checkpoint.py": """def save_checkpoint(network, path):
    temporary = path.with_suffix('.pending')
    network.save(temporary)
    temporary.replace(path)

def load_checkpoint(path):
    network = WellNetwork.load(path)
    return network
""",
    "render.py": """def render_frame(canvas, points, color):
    canvas.clear()
    canvas.begin_path()
    for point in points:
        canvas.line_to(point.x, point.y)
    canvas.stroke(color)
    canvas.present()
""",
    "tokenizer.py": """def tokenize_identifier(identifier):
    identifier = split_camel_case(identifier)
    tokens = []
    for token in identifier.split('_'):
        if len(token) > 1:
            tokens.append(token.lower())
    return tokens
""",
}
QUERIES = [
    (
        "checkpoint.py",
        "def restore_checkpoint(path):\n    network = load_checkpoint(path)\n    return network\n",
    ),
    (
        "render.py",
        "def draw_frame(canvas, points):\n    canvas.begin_path()\n    render_frame(canvas, points, color='green')\n",
    ),
    (
        "tokenizer.py",
        "def split_identifier(identifier):\n    tokens = tokenize_identifier(identifier)\n    return [token.lower() for token in tokens]\n",
    ),
]


def run():
    matrix = learn_byte_embedding(HUNKS.values(), dim=32)
    source = LookupSource.bytes(matrix, window=8)
    index = SignatureIndex(32)
    for key, code in HUNKS.items():
        index.add(key, source.project(code), identifiers=code_tokens(code))
    correct = 0
    for expected, query in QUERIES:
        hits = index.search(
            source.project(query), identifiers=code_tokens(query), lexical_weight=0.35
        )
        hit = hits[0]
        correct += hit.key == expected
        print(
            f"Expected {expected}; retrieved {hit.key}: geometry={hit.geometry:.4f}, "
            f"identifier overlap={hit.identifier_overlap:.4f}, blended={hit.score:.4f}"
        )
        print("Evidence:", HUNKS[hit.key].splitlines()[0])
    assert correct == len(QUERIES)
    print(
        f"{correct}/{len(QUERIES)} authored query/hunk pairs; a small executable example, not a general retrieval benchmark"
    )
    return index


if __name__ == "__main__":
    run()
