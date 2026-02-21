import json
from pantograph.server import Server

def build_so_tactic(
      close: list[str] = None,   # no-arg: constructor, trivial, ring
      hyp:   list[str] = None,   # local-decl: apply, exact, cases
      var:   list[str] = None,   # fresh-var: intro, rintro
      func:  list[str] = None,   # apply congrArg <f>: Nat.succ, etc.
  ) -> str:
      config = {}
      if close: config["close"] = close
      if hyp:   config["hyp"]   = hyp
      if var:   config["var"]   = var
      if func:  config["func"]  = func
      # Lean string literal: double-quoted with inner quotes escaped
      inner = json.dumps(config).replace('"', '\\"')
      return f'so "{inner}"'

oracle_tactic = build_so_tactic(close=["constructor"], 
                      hyp=["induct_rename", "apply", "cases"],
                      var=["intro"],
                      func=["Nat.succ"])


server = Server(
    project_path="/Users/nelsmartin/Lean/RL-Testbed",
    imports=["Init", "RLTestbed.ParameterizedTactic"],
)

goal_expr = "∀ n : Nat, 0 + n = n"
initial_state = server.goal_start(goal_expr)
initial_state = server.goal_tactic(initial_state, "intro h")



suggestion_state = server.goal_tactic(initial_state, oracle_tactic)
for msg in suggestion_state.messages:
        data = json.loads(msg.data)
        for move in data.get("nextMoves", []):
            print(move)
