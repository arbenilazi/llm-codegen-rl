import re
import ast
import hashlib

VARIANTS = [
    # Core control-flow styles
    "use a for-loop",
    "use a while-loop",
    "use reversed iteration",
    "use early returns",
    "avoid early returns",
    "index-based iteration",
    "use enumerate()",
    "avoid enumerate()",

    # Functional / declarative styles
    "functional style with map/filter",
    "list comprehension style",
    "use generator expressions",

    # Data-structure driven variants
    "use a dictionary-based approach",
    "use a set-based approach",
    "avoid extra data structures",
    "use frequency counting",
    "use grouping logic",

    # Algorithmic pattern families
    "sorting + scanning approach",
    "hashing/signature-based approach",
    "two-pointer scanning (if applicable)",
    "split-and-merge approach",
    "recursive divide-and-conquer",
    "iterative dynamic programming",
    "mathematical/closed-form approach (if applicable)",
    "simulation-style approach",

    # Structural layout
    "use a helper function",
    "keep everything in one function",
    "use tuple unpacking where natural",
    "avoid tuple unpacking",
    "avoid slicing",
    "use slicing where natural",

    # Imperative accumulation patterns
    "explicit mutable accumulator",
    "manual stack-based iteration",
]


def synthesize_problem_description(task: dict) -> str:
    """Builds a concise, example-aware problem description."""
    desc = (task.get("problem_description") or "")
    inp = (task.get("input") or "")
    if len(desc) < 200 or len(inp) > len(desc):
        code_blocks = re.findall(r"```(?:python)?(.*?)```", inp, flags=re.S)
        snippets = [blk.strip() for blk in code_blocks[:2] if blk.strip()]
        desc = desc.strip()
        if snippets:
            example_section = "\nExamples:\n" + "\n\n".join(snippets)
            desc = (desc + "\n" + example_section).strip()

    desc = desc.replace("\r", "")
    desc = re.sub(r"[ \t]+", " ", desc)
    desc = re.sub(r"\n{3,}", "\n\n", desc).strip()

    def _trim(text: str, limit: int = 350) -> str:
        if len(text) <= limit:
            return text
        snippet = text[:limit]
        for boundary in [".", "!", "?", "\n"]:
            idx = snippet.rfind(boundary)
            if idx != -1:
                return snippet[: idx + 1].strip()
        return snippet.strip()

    desc = _trim(desc, 350)
    if desc and not desc.endswith((".", "!", "?", "```")):
        desc = desc.rstrip(".!?`") + "."

    return desc


def normalize_code_output(output):
    """Normalizes predictable structural variations in model outputs."""
    try:
        obj = ast.literal_eval(output)
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], list):
            return obj[0]
        return obj
    except Exception:
        return output


def equivalent(a, b):
    """General equivalence check for evaluation tolerance."""
    if a == b:
        return True
    if isinstance(a, list) and isinstance(b, list):
        return sorted(a) == sorted(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-6
    return False


def canonical_signature(code: str) -> str:
    """
    Computes a moderate canonical signature:
    - removes comments and whitespace noise
    - normalizes all variable names (v0, v1, ...)
    - preserves function names and control structure
    - hashes the normalized AST for robust comparison
    """
    # Step 1: remove comments and collapse whitespace.
    cleaned = re.sub(r"#.*", "", code or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Step 2: attempt AST normalization.
    try:
        tree = ast.parse(cleaned)
    except Exception:
        # Fallback: hash the cleaned text
        return hashlib.md5(cleaned.encode()).hexdigest()

    # Step 3: normalize variable names.
    class VarNormalizer(ast.NodeTransformer):
        def __init__(self):
            super().__init__()
            self.map = {}
            self.counter = 0

        def _get_name(self, original):
            if original not in self.map:
                self.map[original] = f"v{self.counter}"
                self.counter += 1
            return self.map[original]

        # Normalize variable identifiers
        def visit_Name(self, node):
            # Do NOT rename function names or attributes
            if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
                return ast.copy_location(
                    ast.Name(id=self._get_name(node.id), ctx=node.ctx),
                    node
                )
            return node

    normalizer = VarNormalizer()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)

    # Step 4: dump normalized AST.
    dumped = ast.dump(normalized_tree, annotate_fields=False, include_attributes=False)

    # Step 5: hash normalized result.
    return hashlib.md5(dumped.encode()).hexdigest()


def build_difficulty_prompt(problem_description: str) -> list[dict]:
    """Builds a short prompt asking the model to classify difficulty from full problem context."""
    text = (
        "You are an expert Python instructor.\n"
        "Classify the difficulty of the following programming problem as Easy, Medium, or Hard.\n"
        "Consider how challenging it is for an intermediate programmer.\n"
        "Respond with only one word: Easy, Medium, or Hard.\n\n"
        f"Problem:\n{problem_description.strip()}"
    )
    return [
        {"role": "system", "content": "You are a helpful expert assistant."},
        {"role": "user", "content": text},
    ]


def build_strategy_prompt(problem_context: str) -> str:
    variants_list = "\n".join(f"- {v}" for v in VARIANTS)

    return (
        "You are an expert Python programmer.\n"
        "Analyze the problem and list several DISTINCT natural ways it can be solved.\n\n"

        "There are two types of diversity:\n"
        "1. **Algorithmic diversity** — using different standard approaches "
        "(e.g., sorting vs hashing vs two-pointer scanning, recursion vs iteration, "
        "frequency counting vs signature hashing, dynamic programming vs greedy, "
        "closed-form math vs simulation, etc.).\n"
        "2. **Structural diversity** — using different code structures while keeping "
        "the SAME algorithm (e.g., for/while loops, helper functions, slicing, "
        "enumerate, accumulators, etc.).\n\n"

        "Your goal:\n"
        "- Propose the variants below ONLY if they are *natural* fits for this task.\n"
        "- Prefer **algorithmic diversity first** when the problem allows multiple true algorithms.\n"
        "- If only one real algorithm exists, then propose several structural variants.\n"
        "- Do NOT propose meaningless or forced variations.\n"
        "- Each variant must be concise and descriptive—one line per variant.\n\n"

        "Here is the catalogue of possible structural/algorithmic ideas:\n"
        f"{variants_list}\n\n"

        "Return ONLY the applicable variants, one per line."
        "\n\nProblem:\n"
        f"{problem_context.strip()}"
    )


