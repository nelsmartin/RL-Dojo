from pantograph.server import Server
from collections import deque
import json

server = Server(
    project_path="/Users/nelsmartin/Lean/PantographTest",
    imports=["Init", "PantographTest.MyTactic"]
)
# ∀ n : Nat, 0 + n = n

goal = "∀ m n : Nat, m + n = 0 → m = 0"
initial_state = server.goal_start(goal)

# BFS queue: (goal_state, tactics_so_far)
queue = deque([(initial_state, [])])
max_depth = 20

while queue:
    state, tactics = queue.popleft()

    if state.is_solved:
        print(f"Proof found: {tactics}")
        break

    if len(tactics) >= max_depth:
        continue

    # Query "so" for suggested next moves (does not alter state)
    suggestion_state = server.goal_tactic(state, "so")
    for msg in suggestion_state.messages:
        data = json.loads(msg.data)
        for move in data.get("nextMoves", []):
            tactic = move["tactic"]
            try:
                next_state = server.goal_tactic(state, tactic)
                queue.append((next_state, tactics + [tactic]))
            except Exception:
                continue

else:
    print("Search exhausted, no proof found.")
