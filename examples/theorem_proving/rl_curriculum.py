"""

approach from connection_test.py with an
RL-trained policy network that learns to rank/select tactics from the `so` oracle.

Flow:
1) Load theorems from a Lean file via LeanDojo.
2) For each epoch, iterate over theorems:
   - Collect a trajectory using the current policy (epsilon-greedy).
   - If proof succeeds, update policy with positive reward.
   - If proof fails, update policy with zero reward.
3) Track success rate over epochs to verify learning.
"""

import argparse
import json
from collections import deque

from pantograph.server import Server
from pantograph.message import TacticFailure, ServerError

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.lean_agent.generator.rl_generator import (
    RLTacticGenerator,
    TacticScorer,
    Trajectory,
)


# ---- Theorem loading (from connection_test.py) ----


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
    """Extract (full_name, theorem_statement) pairs sorted by file position, deduplicated."""
    theorems = []
    for repo in db_dict.get("repositories", []):
        for thm in repo.get("proven_theorems", []):
            name = thm["full_name"]
            stmt = thm["theorem_statement"]
            start = thm["start"]
            theorems.append((name, stmt, start))
    theorems.sort(key=lambda t: eval(t[2]))
    seen = set()
    result = []
    for name, stmt, _ in theorems:
        if name not in seen:
            seen.add(name)
            result.append((name, stmt))
    return result


def goal_expr_from_statement(stmt):
    """Convert a theorem statement to just the type expression after the colon."""
    colon_idx = stmt.index(":")
    expr = stmt[colon_idx + 1 :].strip()
    if expr.endswith(":="):
        expr = expr[:-2].strip()
    return expr


def register_proof(server, name, goal_expr, solved_state, proved_theorems, server_registered):
    """Register a proved theorem in the Lean environment and track it."""
    if name in server_registered:
        # Already in server env from a prior epoch; just ensure it's in the tactic list.
        if name not in proved_theorems:
            proved_theorems.append(name)
        return

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
            server_registered.add(name)
        except ServerError as e:
            print(f"  Warning: could not add {name} to environment: {e}")
    else:
        print(f"  Warning: could not extract proof term for {name}")
    proved_theorems.append(name)


# ---- BFS baseline for comparison ----


def get_candidate_tactics(server, state, proved_theorems):
    """Get candidate tactics from `so` + previously proved theorems."""
    tactics = []
    try:
        suggestion_state = server.goal_tactic(state, "so")
        for msg in suggestion_state.messages:
            try:
                data = json.loads(msg.data)
                for move in data.get("nextMoves", []):
                    tactics.append(move["tactic"])
            except (json.JSONDecodeError, KeyError):
                continue
    except (TacticFailure, ServerError):
        pass

    for thm_name in proved_theorems:
        tactics.append(f"apply {thm_name}")

    return tactics


def prove_with_bfs(server, goal_expr, proved_theorems, max_depth=11):
    """BFS baseline prover."""
    initial_state = server.goal_start(goal_expr)
    queue = deque([(initial_state, [])])

    while queue:
        state, tactics = queue.popleft()
        if state.is_solved:
            return (tactics, state)
        if len(tactics) >= max_depth:
            continue

        candidates = get_candidate_tactics(server, state, proved_theorems)
        for tactic in candidates:
            try:
                next_state = server.goal_tactic(state, tactic)
                queue.append((next_state, tactics + [tactic]))
            except (TacticFailure, ServerError):
                continue

    return None


# ---- RL Training Loop ----


def train_rl_curriculum(
    theorems_path: str,
    commit: str,
    project_path: str,
    imports: list,
    num_epochs: int = 10,
    max_depth: int = 11,
    lr: float = 1e-4,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.05,
    save_path: str = "rl_scorer.pt",
):
    """Main RL training loop."""

    # Load theorems
    print("=== Loading theorems from LeanDojo ===")
    db = load_theorems(theorems_path, commit)
    theorems = extract_theorem_list(db)
    print(f"Found {len(theorems)} theorems: {[name for name, _ in theorems]}\n")

    # Initialize server
    print("=== Initializing Pantograph server ===")
    server = Server(project_path=project_path, imports=imports)
    print("Server ready.\n")

    # Tracks names successfully env_add'd to the server (persists across epochs)
    server_registered: set = set()

    # Initialize RL generator
    scorer = TacticScorer()
    rl_gen = RLTacticGenerator(
        server=server,
        scorer=scorer,
        lr=lr,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    # Track metrics
    epoch_stats = []

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"=== Epoch {epoch + 1}/{num_epochs} (epsilon={rl_gen.epsilon:.3f}) ===")
        print(f"{'='*60}")

        epoch_proved = 0
        epoch_total = len(theorems)
        epoch_loss = 0.0
        epoch_updates = 0

        for name, stmt in theorems:
            goal_expr = goal_expr_from_statement(stmt)
            print(f"\n  [{name}] goal: {goal_expr}")

            # Collect trajectory using current policy
            trajectory = rl_gen.prove_with_policy(
                goal_expr=goal_expr,
                max_depth=max_depth,
            )

            if trajectory.success:
                # Positive reward
                loss = rl_gen.update(trajectory, reward=1.0)
                epoch_proved += 1
                epoch_loss += loss
                epoch_updates += 1
                print(f"    PROVED in {len(trajectory.steps)} steps (loss={loss:.4f})")

                # Register the proof so later theorems can use it
                # Re-prove with BFS to get a proper solved_state for registration
                result = prove_with_bfs(
                    server, goal_expr, rl_gen.proved_theorems, max_depth
                )
                if result:
                    _, solved_state = result
                    register_proof(
                        server, name, goal_expr, solved_state, rl_gen.proved_theorems, server_registered
                    )
            else:
                # Zero reward for failed attempts
                loss = rl_gen.update(trajectory, reward=0.0)
                epoch_loss += loss
                epoch_updates += 1
                print(
                    f"    FAILED after {len(trajectory.steps)} steps (loss={loss:.4f})"
                )

        # Decay exploration
        rl_gen.decay_epsilon()

        # Epoch summary
        success_rate = epoch_proved / epoch_total if epoch_total > 0 else 0.0
        avg_loss = epoch_loss / epoch_updates if epoch_updates > 0 else 0.0
        stats = {
            "epoch": epoch + 1,
            "proved": epoch_proved,
            "total": epoch_total,
            "success_rate": success_rate,
            "avg_loss": avg_loss,
            "epsilon": rl_gen.epsilon,
        }
        epoch_stats.append(stats)

        print(f"\n  Epoch {epoch + 1} summary:")
        print(f"    Proved: {epoch_proved}/{epoch_total} ({success_rate:.1%})")
        print(f"    Avg loss: {avg_loss:.4f}")
        print(f"    Epsilon: {rl_gen.epsilon:.3f}")

    # Save model
    rl_gen.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("=== Training Complete ===")
    print(f"{'='*60}")
    print(f"\nEpoch | Proved | Rate   | Avg Loss | Epsilon")
    print(f"------+--------+--------+----------+--------")
    for s in epoch_stats:
        print(
            f"  {s['epoch']:3d}  | {s['proved']:4d}   | {s['success_rate']:.1%}  | {s['avg_loss']:.4f}  | {s['epsilon']:.3f}"
        )

    # Run BFS baseline for comparison
    print(f"\n{'='*60}")
    print("=== BFS Baseline (for comparison) ===")
    print(f"{'='*60}")
    bfs_proved = []
    for name, stmt in theorems:
        goal_expr = goal_expr_from_statement(stmt)
        result = prove_with_bfs(server, goal_expr, bfs_proved, max_depth)
        if result:
            _, solved_state = result
            register_proof(server, name, goal_expr, solved_state, bfs_proved, server_registered)
            print(f"  BFS proved {name}")
        else:
            print(f"  BFS failed {name}")

    print(f"\nBFS baseline: {len(bfs_proved)}/{len(theorems)} proved")
    if epoch_stats:
        final = epoch_stats[-1]
        print(f"RL final:     {final['proved']}/{final['total']} proved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Curriculum Theorem Prover")
    parser.add_argument(
        "--theorems-path",
        default="/Users/nelsmartin/Lean/RL-Testbed",
        help="Path to the Lean project with theorems",
    )
    parser.add_argument(
        "--commit",
        default="a5833359eb17ac329c4392702bd226cf4cc29771",
        help="Git commit to trace",
    )
    parser.add_argument(
        "--project-path",
        default="/Users/nelsmartin/Lean/RL-Testbed",
        help="Path to the Lean project for Pantograph",
    )
    parser.add_argument(
        "--imports",
        nargs="+",
        default=["Init", "RLTestbed.MyTactic"],
        help="Lean imports for Pantograph",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max-depth", type=int, default=11, help="Max proof search depth")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--save-path", default="rl_scorer.pt", help="Path to save model")

    args = parser.parse_args()

    train_rl_curriculum(
        theorems_path=args.theorems_path,
        commit=args.commit,
        project_path=args.project_path,
        imports=args.imports,
        num_epochs=args.epochs,
        max_depth=args.max_depth,
        lr=args.lr,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        save_path=args.save_path,
    )
