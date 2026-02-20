"""RL-based tactic generator that scores/ranks candidate tactics from the `so` oracle."""

import json
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.optim import Adam

from pantograph.message import TacticFailure, ServerError

from lean_dojo_v2.lean_agent.generator.model import TacticGenerator
from lean_dojo_v2.lean_dojo import Pos


@dataclass
class Trajectory:
    """A single proving trajectory (rollout)."""

    steps: List[Tuple[str, str, torch.Tensor]] = field(default_factory=list)
    # Each step: (state_str, chosen_tactic, log_prob)
    success: bool = False


class TacticScorer(nn.Module):
    """Scores (state, tactic) pairs using a sentence encoder + MLP head."""

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        # Freeze the sentence encoder — only train the MLP head
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_str: str, tactic_strs: List[str]) -> torch.Tensor:
        """Score each tactic given the current proof state.

        Returns a tensor of shape (len(tactic_strs),) with raw scores.
        """
        if not tactic_strs:
            return torch.tensor([])

        with torch.no_grad():
            state_emb = self.encoder.encode(
                [state_str], convert_to_tensor=True
            ).cpu()  # (1, embed_dim)
            tactic_embs = self.encoder.encode(
                tactic_strs, convert_to_tensor=True
            ).cpu()  # (N, embed_dim)

        # Expand state embedding to match number of tactics
        state_expanded = state_emb.expand(len(tactic_strs), -1)  # (N, embed_dim)
        combined = torch.cat([state_expanded, tactic_embs], dim=-1)  # (N, 2*embed_dim)
        scores = self.head(combined).squeeze(-1)  # (N,)
        return scores


class RLTacticGenerator(TacticGenerator):
    """Tactic generator that uses an RL-trained policy to rank candidates from `so`."""

    def __init__(
        self,
        server,
        scorer: Optional[TacticScorer] = None,
        lr: float = 1e-4,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        gamma: float = 0.99,
    ):
        self.server = server
        self.scorer = scorer or TacticScorer()
        self.optimizer = Adam(
            self.scorer.head.parameters(), lr=lr
        )
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.proved_theorems: List[str] = []

    def get_candidate_tactics(self, state, proved_theorems: Optional[List[str]] = None) -> List[str]:
        """Get candidate tactics from `so` oracle + previously proved theorems."""
        if proved_theorems is None:
            proved_theorems = self.proved_theorems
        tactics = []

        try:
            suggestion_state = self.server.goal_tactic(state, "so")
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

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        """Generate tactic candidates scored by the policy network.

        Conforms to the TacticGenerator interface for use with BestFirstSearchProver.
        """
        # state here is a string representation; we need a goal state object for `so`
        # In the BestFirstSearchProver context, we'd need the actual goal state.
        # For now, return scored candidates if we have them cached, or empty list.
        return self._score_tactics(state, num_samples)

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, t, p, num_samples)
            for s, f, t, p in zip(state, file_path, theorem_full_name, theorem_pos)
        ]

    def _score_tactics(self, state_str: str, num_samples: int) -> List[Tuple[str, float]]:
        """Score candidate tactics with the policy network."""
        candidates = self.get_candidate_tactics_from_str(state_str)
        if not candidates:
            return []

        self.scorer.eval()
        with torch.no_grad():
            scores = self.scorer(state_str, candidates)
            log_probs = F.log_softmax(scores, dim=0)

        # Sort by score descending, return top num_samples
        indexed = list(zip(candidates, log_probs.tolist()))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:num_samples]

    def get_candidate_tactics_from_str(self, state_str: str) -> List[str]:
        """Fallback: return apply candidates from proved theorems when we only have string state."""
        tactics = []
        for thm_name in self.proved_theorems:
            tactics.append(f"apply {thm_name}")
        return tactics

    def select_action(
        self, state_str: str, candidates: List[str]
    ) -> Tuple[str, float, torch.Tensor]:
        """Epsilon-greedy action selection.

        Returns (chosen_tactic, score, log_prob_tensor).
        """
        if not candidates:
            raise ValueError("No candidate tactics available")

        self.scorer.eval()
        scores = self.scorer(state_str, candidates)
        log_probs = F.log_softmax(scores, dim=0)
        probs = F.softmax(scores, dim=0)

        if random.random() < self.epsilon:
            # Explore: pick a random tactic
            idx = random.randrange(len(candidates))
        else:
            # Exploit: pick the highest-scoring tactic
            idx = torch.argmax(scores).item()

        chosen_tactic = candidates[idx]
        score = scores[idx].item()
        log_prob = log_probs[idx]

        return chosen_tactic, score, log_prob

    def update(self, trajectory: Trajectory, reward: float) -> float:
        """REINFORCE policy gradient update.

        Args:
            trajectory: The collected trajectory of (state, tactic, log_prob) tuples.
            reward: 1.0 for successful proof, 0.0 for failure.

        Returns:
            The loss value.
        """
        if not trajectory.steps:
            return 0.0

        self.scorer.train()
        self.optimizer.zero_grad()

        # Compute discounted returns for each step
        n_steps = len(trajectory.steps)
        returns = []
        G = reward
        for i in range(n_steps - 1, -1, -1):
            returns.insert(0, G)
            G *= self.gamma

        # REINFORCE loss: -log_prob * return
        loss = torch.tensor(0.0, requires_grad=True)
        for (state_str, tactic, log_prob), G_t in zip(trajectory.steps, returns):
            loss = loss + (-log_prob * G_t)

        loss = loss / n_steps  # normalize by trajectory length

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def prove_with_policy(
        self,
        goal_expr: str,
        max_depth: int = 11,
    ) -> Trajectory:
        """Single rollout using the current policy (epsilon-greedy).

        Returns a Trajectory with the sequence of (state, tactic, log_prob) steps.
        """
        trajectory = Trajectory()

        try:
            state = self.server.goal_start(goal_expr)
        except ServerError:
            return trajectory

        for _ in range(max_depth):
            if state.is_solved:
                trajectory.success = True
                return trajectory

            # Get candidates from `so` + proved theorems
            candidates = self.get_candidate_tactics(state, self.proved_theorems)
            if not candidates:
                return trajectory

            state_str = str(state)

            # Select action with epsilon-greedy
            chosen_tactic, score, log_prob = self.select_action(state_str, candidates)
            trajectory.steps.append((state_str, chosen_tactic, log_prob))

            # Apply the chosen tactic
            try:
                state = self.server.goal_tactic(state, chosen_tactic)
            except (TacticFailure, ServerError):
                return trajectory

        # Check if we solved it at the end
        if state.is_solved:
            trajectory.success = True

        return trajectory

    def save(self, path: str):
        """Save the scorer's MLP head weights."""
        torch.save({
            "head_state_dict": self.scorer.head.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        """Load the scorer's MLP head weights."""
        checkpoint = torch.load(path, weights_only=True)
        self.scorer.head.load_state_dict(checkpoint["head_state_dict"])
        self.epsilon = checkpoint["epsilon"]
