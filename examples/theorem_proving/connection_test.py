from pantograph.server import Server
from pantograph.message import TacticFailure, ServerError
from lean_dojo_v2.database import DynamicDatabase
from collections import deque
import json


def build_so_tactic(
    close: list[str] = None,   # no-arg: constructor, trivial, rfl
    hyp:   list[str] = None,   # local-decl: apply, exact, cases
    var:   list[str] = None,   # fresh-var: intro, rintro
    func:  list[str] = None,   # apply congrArg <f>: Nat.succ, etc.
) -> str:
    config = {}
    if close: config["close"] = close
    if hyp:   config["hyp"]   = hyp
    if var:   config["var"]   = var
    if func:  config["func"]  = func
    inner = json.dumps(config).replace('"', '\\"')
    return f'so "{inner}"'

"""
Curriculum Theorem Prover

Flow:
1) Load theorems from a Lean file via LeanDojo.
2) For each theorem in order:
    Try to prove it with BFS, keeping track of what you proved.
3) If any unsolved theorems remain, try again to solve them, this time
   using the solved theorems as apply candidates.
"""

theorems_path = "/Users/nelsmartin/Lean/RL-Testbed"
commit = "4de4d162bc17cf0a530487a471e780457be55849"


def load_theorems(path, commit):
    """Load and trace theorems from a Lean project."""
    database = DynamicDatabase()
    repo = database.trace_repository(
        url=path,
        commit=commit,
        build_deps=False,
    )
    if repo:
        database.add_repository(repo)
    return database.to_dict()


def extract_theorem_list(db_dict):
    """Extract (full_name, theorem_statement) pairs from the database dict.

    Returns theorems sorted by their start position to preserve file order.
    """
    theorems = []
    for repo in db_dict.get("repositories", []):
        for thm in repo.get("proven_theorems", []):
            name = thm["full_name"]
            stmt = thm["theorem_statement"]
            start = thm["start"]  # e.g. "(4, 1)"
            theorems.append((name, stmt, start))
    # Sort by start position to get file order
    theorems.sort(key=lambda t: eval(t[2]))
    return [(name, stmt) for name, stmt, _ in theorems]


def get_candidate_tactics(server, state, proved_theorems, oracle_tactic):
    """Get candidate tactics from the parameterized oracle plus
    apply calls for previously proved theorems.

    Returns a list of tactic strings.
    """
    tactics = []

    # Get suggestions from the parameterized 'so' oracle
    try:
        suggestion_state = server.goal_tactic(state, oracle_tactic)
        for msg in suggestion_state.messages:
            try:
                data = json.loads(msg.data)
                for move in data.get("nextMoves", []):
                    tactics.append(move["tactic"])
            except (json.JSONDecodeError, KeyError):
                continue
    except (TacticFailure, ServerError):
        pass

    # Add apply candidates from previously proved theorems
    for thm_name in proved_theorems:
        tactics.append(f"apply {thm_name}")

    return tactics


def prove_with_bfs(server, goal_expr, proved_theorems, oracle_tactic, max_depth=15):
    """Try to prove goal_expr using BFS.

    Returns (tactics_list, solved_state) if proved, or None if not.
    """
    initial_state = server.goal_start(goal_expr)

    # BFS queue: (goal_state, tactics_so_far)
    queue = deque([(initial_state, [])])

    while queue:
        state, tactics = queue.popleft()

        if state.is_solved:
            return (tactics, state)

        if len(tactics) >= max_depth:
            continue

        candidates = get_candidate_tactics(server, state, proved_theorems, oracle_tactic)

        for tactic in candidates:
            try:
                next_state = server.goal_tactic(state, tactic)
                queue.append((next_state, tactics + [tactic]))
            except (TacticFailure, ServerError):
                continue

    return None


def register_proof(server, name, goal_expr, solved_state, proved_theorems):
    """Register a proved theorem in the Lean environment and track it."""
    proof = server.goal_root(solved_state)
    if proof is not None:
        try:
            server.env_add(
                name=name,
                levels=[],
                t=goal_expr,
                v=proof,
                is_theorem=True,
            )
        except ServerError as e:
            print(f"  Warning: could not add {name} to environment: {e}")
    else:
        print(f"  Warning: could not extract proof term for {name}")
    proved_theorems.append(name)


def goal_expr_from_statement(stmt):
    """Convert a theorem statement like 'theorem t1 : ∀ n : Nat, ...'
    to just the type expression after the colon.

    Strips 'theorem <name> :' prefix and ' :=' suffix.
    """
    # Remove 'theorem <name> : ' prefix
    colon_idx = stmt.index(":")
    expr = stmt[colon_idx + 1:].strip()
    # Remove trailing ':=' if present
    if expr.endswith(":="):
        expr = expr[:-2].strip()
    return expr


# ---- Main ----

print("=== Loading theorems from LeanDojo ===")
db = load_theorems(theorems_path, commit)
theorems = extract_theorem_list(db)
print(f"Found {len(theorems)} theorems: {[name for name, _ in theorems]}\n")

print("=== Initializing Pantograph server ===")
server = Server(
    project_path="/Users/nelsmartin/Lean/RL-Testbed",
    imports=["Init", "RLTestbed.ParameterizedTactic"],
)
print("Server ready.\n")

oracle_tactic = build_so_tactic(
    close=["constructor"],
    hyp=["induct_rename", "apply", "cases"],
    var=["intro"],
    func=["Nat.succ"],
)

# oracle_tactic = build_so_tactic(
#     close=["simp"]
# )
print(f"Oracle tactic: {oracle_tactic}\n")

proved_theorems = []  # list of theorem names added to the environment
unsolved = []

print("=== Proving theorems in order ===")
for name, stmt in theorems:
    goal_expr = goal_expr_from_statement(stmt)
    print(f"Attempting {name}: {goal_expr}")

    result = prove_with_bfs(server, goal_expr, proved_theorems, oracle_tactic)

    if result:
        tactics, solved_state = result
        print(f"  Proved! Tactics: {tactics}")
        register_proof(server, name, goal_expr, solved_state, proved_theorems)
    else:
        print(f"  Could not prove.")
        unsolved.append(name)

print(f"\n=== Summary ===")
print(f"Proved: {proved_theorems}")
if unsolved:
    print(f"Unsolved: {unsolved}")
