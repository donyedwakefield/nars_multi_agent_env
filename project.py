from __future__ import annotations


import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Matplotlib is optional so the sim still runs on a bare Python install.
_PLOT_IMPORT_ERROR: Optional[str] = None
try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as exc:  # ImportError, bad backend, cache dir permissions, etc.
    plt = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _PLOT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _plotting_available() -> bool:
    return plt is not None and np is not None


def _plot_output_path(filename: str) -> str:
    """PNG path next to this file (cwd-independent)."""
    return str(Path(__file__).resolve().parent / filename)


# Facts — toy environment

# Vocabulary for the small demo generator; cycles if you ask for more facts than slots.
_SUBJECTS = (
    "Object A",
    "Object B",
    "Object C",
    "Object D",
    "Signal 1",
    "Signal 2",
    "Signal 3",
    "Signal 4",
)

_PROPERTIES = (
    "is active",
    "is inactive",
    "is red",
    "is blue",
    "is open",
    "is closed",
    "is safe",
    "is unsafe",
)


@dataclass(frozen=True)
class Fact:
    """One proposition and whether it holds in the env."""

    fact_id: str
    proposition: str
    truth: bool


def generate_environment_facts(num_facts: int, seed: Optional[int] = None) -> List[Fact]:
    """Small subject/property lists — what `main()` uses for the short console demo."""
    rng = random.Random(seed)
    facts: List[Fact] = []
    used_propositions: set[str] = set()

    for i in range(num_facts):
        subject = _SUBJECTS[i % len(_SUBJECTS)]
        prop = _PROPERTIES[i % len(_PROPERTIES)]
        proposition = f"{subject} {prop}"

        if proposition in used_propositions:
            proposition = f"{proposition} #{i + 1}"

        used_propositions.add(proposition)
        truth = rng.choice([True, False])

        facts.append(
            Fact(
                fact_id=f"fact_{i + 1}",
                proposition=proposition,
                truth=truth,
            )
        )

    return facts


def generate_more_varied_facts(num_facts: int, seed: Optional[int] = None) -> List[Fact]:
    """Larger vocabulary rotation; `run_large_experiment` uses this so longer runs don't look repetitive."""
    rng = random.Random(seed)

    subjects = (
        "Object A", "Object B", "Object C", "Object D",
        "Object E", "Object F", "Object G", "Object H",
        "Signal 1", "Signal 2", "Signal 3", "Signal 4",
        "Signal 5", "Signal 6", "Signal 7", "Signal 8",
    )

    properties = (
        "is active", "is inactive", "is red", "is blue",
        "is open", "is closed", "is safe", "is unsafe",
        "is visible", "is hidden", "is nearby", "is far",
    )

    facts: List[Fact] = []
    used_propositions: set[str] = set()

    for i in range(num_facts):
        subject = subjects[i % len(subjects)]
        prop = properties[(i * 3) % len(properties)]
        proposition = f"{subject} {prop}"

        if proposition in used_propositions:
            proposition = f"{proposition} #{i + 1}"

        used_propositions.add(proposition)
        truth = rng.choice([True, False])

        facts.append(
            Fact(
                fact_id=f"fact_{i + 1}",
                proposition=proposition,
                truth=truth,
            )
        )

    return facts


# Statements — who said what about which fact

@dataclass(frozen=True)
class Statement:
    """One timestep, one agent, one fact: what they claimed vs what is actually true."""

    step: int
    source_agent: str
    fact_id: str
    proposition: str
    claimed_truth: bool
    actual_truth: bool

    def to_dict(self) -> Dict[str, object]:
        """JSON-ish dict for logs."""
        return {
            "step": self.step,
            "source_agent": self.source_agent,
            "fact_id": self.fact_id,
            "proposition": self.proposition,
            "claimed_truth": self.claimed_truth,
            "actual_truth": self.actual_truth,
            "was_correct": self.claimed_truth == self.actual_truth,
        }

    def to_text(self) -> str:
        """One line for the terminal."""
        claim_label = "true" if self.claimed_truth else "false"
        actual_label = "true" if self.actual_truth else "false"
        return (
            f"[t={self.step}] {self.source_agent} -> "
            f"'{self.proposition}' "
            f"(claimed={claim_label}, actual={actual_label})"
        )

    @staticmethod
    def _normalize_symbol(text: str) -> str:
        """Rough cleanup for Narsese-ish identifiers."""
        cleaned = text.strip().replace(" ", "_").replace("'", "").replace('"', "")
        return cleaned

    def _parse_simple_proposition(self) -> Tuple[str, str]:
        """'X is Y' -> subject, attribute; else fallback."""
        text = self.proposition.strip()
        if " is " in text:
            subject, attribute = text.split(" is ", 1)
            return subject.strip(), attribute.strip()
        if " are " in text:
            subject, attribute = text.split(" are ", 1)
            return subject.strip(), attribute.strip()
        return text, "unknown"

    def to_narsese_judgment(self, confidence: float = 0.90) -> str:
        """Single judgment line: what we actually pipe into ONA stdin (no comment prefix)."""
        subject, attribute = self._parse_simple_proposition()
        subject_term = "{" + self._normalize_symbol(subject) + "}"

        if self.claimed_truth:
            predicate_text = self._normalize_symbol(attribute)
        else:
            predicate_text = "not_" + self._normalize_symbol(attribute)

        predicate_term = "[" + predicate_text + "]"
        return f"<{subject_term} --> {predicate_term}>. %1.00;{confidence:.2f}%"

    def to_narsese(self, confidence: float = 0.90) -> str:
        """Human-readable comment plus judgment — nicer in saved .nal files, not sent to ONA."""
        judgment_line = self.to_narsese_judgment(confidence=confidence)
        comment_line = (
            f"' source={self.source_agent} "
            f"fact={self.fact_id} "
            f"step={self.step}"
        )
        return comment_line + "\n" + judgment_line


# Agents

class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    def _statement(self, fact: Fact, step: int, claimed_truth: bool) -> Statement:
        return Statement(
            step=step,
            source_agent=self.name,
            fact_id=fact.fact_id,
            proposition=fact.proposition,
            claimed_truth=claimed_truth,
            actual_truth=fact.truth,
        )

    @abstractmethod
    def make_statement(self, fact: Fact, step: int) -> Statement:
        ...


class TruthfulAgent(BaseAgent):
    """Honest."""

    def make_statement(self, fact: Fact, step: int) -> Statement:
        return self._statement(fact, step, claimed_truth=fact.truth)


class DeceptiveAgent(BaseAgent):
    """Flips the boolean."""

    def make_statement(self, fact: Fact, step: int) -> Statement:
        return self._statement(fact, step, claimed_truth=not fact.truth)


class NoisyAgent(BaseAgent):
    """Coin flip each time."""

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)

    def make_statement(self, fact: Fact, step: int) -> Statement:
        claimed = self._rng.choice([True, False])
        return self._statement(fact, step, claimed_truth=claimed)


class StrategicAgent(BaseAgent):
    """Honest for N steps, then lies for N, repeat."""

    def __init__(self, name: str, switch_interval: int = 3) -> None:
        super().__init__(name)
        self.switch_interval = max(1, switch_interval)

    def make_statement(self, fact: Fact, step: int) -> Statement:
        deceptive_phase = ((step // self.switch_interval) % 2) == 1
        claimed = not fact.truth if deceptive_phase else fact.truth
        return self._statement(fact, step, claimed_truth=claimed)


def build_large_agent_pool(seed: Optional[int] = None) -> List[BaseAgent]:
    """Fixed roster that lines up with `_true_reliability_map` agent names."""
    rng = random.Random(seed)

    agents: List[BaseAgent] = []

    for i in range(3):
        agents.append(TruthfulAgent(f"truthful_{i + 1}"))

    for i in range(3):
        agents.append(DeceptiveAgent(f"deceptive_{i + 1}"))

    for i in range(3):
        agents.append(NoisyAgent(f"noisy_{i + 1}", seed=rng.randint(1, 10_000)))

    for i, interval in enumerate([2, 3, 4], start=1):
        agents.append(StrategicAgent(f"strategic_{i}", switch_interval=interval))

    return agents


# Simulation — drive the statement stream

class Simulation:
    """Each timestep, every agent picks a random fact and speaks."""

    def __init__(
        self,
        facts: Sequence[Fact],
        agents: Sequence[BaseAgent],
        seed: Optional[int] = None,
    ) -> None:
        if not facts:
            raise ValueError("Need at least one fact.")
        if not agents:
            raise ValueError("Need at least one agent.")

        self.facts = list(facts)
        self.agents = list(agents)
        self._rng = random.Random(seed)

    def run(self, num_steps: int) -> List[Statement]:
        """Every agent speaks once per timestep; fact picked at random."""
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")

        stream: List[Statement] = []
        for step in range(1, num_steps + 1):
            for agent in self.agents:
                fact = self._rng.choice(self.facts)
                stream.append(agent.make_statement(fact, step))
        return stream


# Belief tracker (Python only — not NARS memory)

@dataclass
class BeliefRecord:
    """Tally for one proposition string."""

    proposition: str
    support_count: int = 0
    contradiction_count: int = 0

    def add_observation(self, supported: bool) -> None:
        if supported:
            self.support_count += 1
        else:
            self.contradiction_count += 1

    def confidence(self) -> float:
        total = self.support_count + self.contradiction_count
        if total == 0:
            return 0.5
        return self.support_count / total


@dataclass
class SourceRecord:
    """Per-agent accuracy tally."""

    source_agent: str
    correct: int = 0
    incorrect: int = 0

    def add_observation(self, correct: bool) -> None:
        if correct:
            self.correct += 1
        else:
            self.incorrect += 1

    def reliability(self) -> float:
        total = self.correct + self.incorrect
        if total == 0:
            return 0.5
        return self.correct / total


class BeliefTracker:
    """
    Running tallies from the raw statement list. Useful for charts and sanity checks;
    do not confuse this with ONA's internal graph.
    """

    def __init__(self) -> None:
        self.beliefs: Dict[str, BeliefRecord] = {}
        self.sources: Dict[str, SourceRecord] = {}
        self.history: List[dict] = []

    def update(self, statement: Statement) -> None:
        proposition = statement.proposition
        source = statement.source_agent
        claim_matches_reality = statement.claimed_truth == statement.actual_truth
        # "Support" here means they asserted true and that matched reality (project-specific rule).
        proposition_supported = claim_matches_reality and statement.claimed_truth

        if proposition not in self.beliefs:
            self.beliefs[proposition] = BeliefRecord(proposition=proposition)
        if source not in self.sources:
            self.sources[source] = SourceRecord(source_agent=source)

        self.beliefs[proposition].add_observation(proposition_supported)
        self.sources[source].add_observation(claim_matches_reality)

        self.history.append(
            {
                "step": statement.step,
                "source_agent": source,
                "fact_id": statement.fact_id,
                "proposition": proposition,
                "claimed_truth": statement.claimed_truth,
                "actual_truth": statement.actual_truth,
                "correct": claim_matches_reality,
            }
        )

    def belief_summary(self) -> List[dict]:
        rows = []
        for proposition, record in sorted(self.beliefs.items()):
            rows.append(
                {
                    "proposition": proposition,
                    "support_count": record.support_count,
                    "contradiction_count": record.contradiction_count,
                    "confidence": round(record.confidence(), 3),
                }
            )
        return rows

    def source_summary(self) -> List[dict]:
        rows = []
        for source, record in sorted(self.sources.items()):
            rows.append(
                {
                    "source_agent": source,
                    "correct": record.correct,
                    "incorrect": record.incorrect,
                    "reliability": round(record.reliability(), 3),
                }
            )
        return rows


# Narsese text export (.nal style)


def format_statements_as_narsese_document(
    statements: Sequence[Statement], confidence: float = 0.90
) -> str:
    """Blank-line separated blocks: comment + judgment per statement."""
    return "\n\n".join(s.to_narsese(confidence=confidence) for s in statements)


def save_narsese_experiment_file(
    statements: Sequence[Statement],
    filepath: str,
    confidence: float = 0.90,
) -> None:
    Path(filepath).write_text(
        format_statements_as_narsese_document(statements, confidence),
        encoding="utf-8",
    )


# Metrics (mostly from the Python tracker)

def compute_source_reliability(source_summary: Sequence[dict]) -> List[dict]:
    """R = correct / (correct + wrong)."""
    results = []

    for agent in source_summary:
        correct = agent["correct"]
        incorrect = agent["incorrect"]
        total = correct + incorrect

        reliability = correct / total if total > 0 else 0.5

        results.append(
            {
                "source_agent": agent["source_agent"],
                "correct": correct,
                "incorrect": incorrect,
                "reliability": reliability,
            }
        )

    return results


def compute_belief_accuracy(statements: Sequence[Statement]) -> float:
    """Fraction of statements where claim matches ground truth."""
    if not statements:
        return 0.0

    correct = 0
    for s in statements:
        if s.claimed_truth == s.actual_truth:
            correct += 1

    return correct / len(statements)


def compute_belief_confidence(belief_summary: Sequence[dict]) -> List[dict]:
    """support / (support + contradiction) per proposition."""
    results = []

    for belief in belief_summary:
        support = belief["support_count"]
        contradiction = belief["contradiction_count"]
        total = support + contradiction

        confidence = support / total if total > 0 else 0.5

        results.append(
            {
                "proposition": belief["proposition"],
                "support_count": support,
                "contradiction_count": contradiction,
                "confidence": confidence,
            }
        )

    return results


def compute_trust_inference(source_summary: Sequence[dict], true_reliability_map: Dict[str, float]) -> float:
    """How close estimated agent reliability is to our fixed ground-truth map (1 - mean abs error)."""
    if not source_summary:
        return 0.0

    total_error = 0.0
    count = 0

    for agent in source_summary:
        name = agent["source_agent"]
        estimated = agent["reliability"]
        true_value = true_reliability_map.get(name, 0.5)

        total_error += abs(estimated - true_value)
        count += 1

    average_error = total_error / count if count > 0 else 0.0
    trust_score = 1.0 - average_error

    return trust_score


def _true_reliability_map() -> Dict[str, float]:
    """Known reliabilities for built-in agent names — only used to score trust inference."""
    return {
        "truthful_1": 1.0,
        "truthful_2": 1.0,
        "truthful_3": 1.0,
        "deceptive_1": 0.0,
        "deceptive_2": 0.0,
        "deceptive_3": 0.0,
        "noisy_1": 0.5,
        "noisy_2": 0.5,
        "noisy_3": 0.5,
        "strategic_1": 0.5,
        "strategic_2": 0.5,
        "strategic_3": 0.5,
    }


def compute_misinformation_resistance(statements: Sequence[Statement]) -> float:
    """Fraction of statements where the claim matched ground truth (same formula as belief accuracy here)."""
    if not statements:
        return 0.0

    correct = sum(1 for s in statements if s.claimed_truth == s.actual_truth)
    return correct / len(statements)


def run_all_metrics(
    statements: Sequence[Statement],
    belief_summary: Sequence[dict],
    source_summary: Sequence[dict]
) -> Dict[str, object]:
    """One dict for the console and for matplotlib; ONA runs append nars_* keys later."""
    source_reliability_results = compute_source_reliability(source_summary)
    belief_accuracy = compute_belief_accuracy(statements)
    belief_confidence_results = compute_belief_confidence(belief_summary)

    trust_inference_score = compute_trust_inference(
        source_summary=source_summary,
        true_reliability_map=_true_reliability_map(),
    )

    misinformation_resistance = compute_misinformation_resistance(statements)

    return {
        "source_reliability_results": source_reliability_results,
        "belief_accuracy": belief_accuracy,
        "belief_confidence_results": belief_confidence_results,
        "trust_inference_score": trust_inference_score,
        "misinformation_resistance": misinformation_resistance,
    }


# Matplotlib helpers (Python-side metrics plots)


def _finalize_plot(title, xlabel=None, ylabel=None, save_path=None):
    """Apply labels, save to disk if asked, always close figures so the next plot starts clean."""
    if not _plotting_available():
        return
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        if not out.is_absolute():
            out = Path(__file__).resolve().parent / out
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=200, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()


def visualize_source_reliability_results(source_reliability_results, save_path=None):
    """Bar + reliability line per agent."""
    if not _plotting_available():
        return
    if not source_reliability_results:
        print("No source reliability results to plot.")
        return

    source_reliability_results = sorted(
        source_reliability_results,
        key=lambda x: x["source_agent"]
    )

    agents = [row["source_agent"] for row in source_reliability_results]
    correct = [row["correct"] for row in source_reliability_results]
    incorrect = [row["incorrect"] for row in source_reliability_results]
    reliability = [row["reliability"] for row in source_reliability_results]

    x = np.arange(len(agents))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(x - width / 2, correct, width, label="Correct")
    ax1.bar(x + width / 2, incorrect, width, label="Incorrect")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, rotation=20, ha="right")
    ax1.set_ylabel("Count")

    ax2 = ax1.twinx()
    ax2.plot(x, reliability, marker="o", linewidth=2, label="Reliability")
    ax2.set_ylabel("Reliability")
    ax2.set_ylim(0, 1.05)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center")

    _finalize_plot(
        title="Source Reliability Results",
        xlabel="Agent",
        ylabel="Count / Reliability",
        save_path=save_path
    )


def visualize_all_metrics_image(metrics: Dict[str, object], save_path: Optional[str] = None) -> None:
    """Horizontal bars: headline scalars plus mean Python-tracker confidence across propositions."""
    if not _plotting_available():
        return

    labels: List[str] = []
    vals: List[float] = []

    labels.append("Belief accuracy")
    vals.append(float(metrics.get("belief_accuracy", 0.0)))

    labels.append("Trust inference score")
    vals.append(float(metrics.get("trust_inference_score", 0.0)))

    labels.append("Misinformation resistance")
    vals.append(float(metrics.get("misinformation_resistance", 0.0)))

    bcr = metrics.get("belief_confidence_results") or []
    if isinstance(bcr, list) and bcr:
        mean_conf = sum(float(r["confidence"]) for r in bcr) / len(bcr)
        labels.append("Mean belief confidence (propositions)")
        vals.append(mean_conf)

    y = np.arange(len(labels))
    h = max(4.0, 0.45 * len(labels))
    plt.figure(figsize=(9, h))
    plt.barh(y, vals, color="steelblue")
    plt.yticks(y, labels)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Score (0–1)")
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    _finalize_plot(
        title="All metrics (aggregates)",
        xlabel="Score",
        ylabel=None,
        save_path=save_path,
    )


# ONA stdout parsing + NARS-side metrics

# ONA prints lines like: Input: <...>. Priority=... Stamp=[n] Truth: frequency=..., confidence=...
# We normalize judgment "cores" (subject --> predicate) so we can match log lines back to statements.
# Important: split on " Stamp=" with a space — the format is not ". Stamp=".


def judgment_core_from_statement(st: Statement, confidence: float = 0.90) -> str:
    """Normalized judgment key so ONA log bodies can be string-matched to our inputs."""
    j = st.to_narsese_judgment(confidence=confidence)
    if "%" in j:
        j = j[: j.index("%")].strip()
    return re.sub(r"\s+", "", j)


def core_key_from_ona_body(body: str) -> str:
    """Same normalization as judgment_core_from_statement: strip spaces, ensure trailing dot."""
    b = body.strip()
    if not b.endswith("."):
        b = b + "."
    return re.sub(r"\s+", "", b)


def parse_ona_line(line: str) -> Optional[Dict[str, object]]:
    """Pull structured fields out of one ONA stdout line, or None if it isn't a judgment line."""
    s = line.strip()
    if not s:
        return None
    kind: Optional[str] = None
    rest = ""
    for k in ("Input", "Derived", "Revised", "Selected"):
        pref = f"{k}: "
        if s.startswith(pref):
            kind = k
            rest = s[len(pref) :].strip()
            break
    if kind is None or ". Priority=" not in rest:
        return None
    body, tail = rest.rsplit(". Priority=", 1)
    body = body.strip()
    if " Stamp=" not in tail:
        return None
    priority_s, tail2 = tail.split(" Stamp=", 1)
    if "] Truth:" not in tail2:
        return None
    stamp_raw, truth_raw = tail2.split("] Truth:", 1)
    stamp = stamp_raw.strip() + "]"
    truth_raw = truth_raw.strip()
    if "frequency=" not in truth_raw or "confidence=" not in truth_raw:
        return None
    fr_part, co_part = truth_raw.split("confidence=", 1)
    freq_s = fr_part.split("frequency=", 1)[1].strip().rstrip(",")
    conf_s = co_part.strip()
    try:
        priority = float(priority_s.strip())
        frequency = float(freq_s)
        confidence = float(conf_s)
    except ValueError:
        return None
    return {
        "kind": kind,
        "body": body,
        "priority": priority,
        "stamp": stamp,
        "frequency": frequency,
        "confidence": confidence,
    }


def judgment_polarity_positive(core: str) -> Optional[bool]:
    """Whether the predicate (ignoring not_) encodes 'positive' polarity for frequency>=0.5."""
    core = core.rstrip(".").strip()
    if "-->" not in core:
        return None
    rhs = core.split("-->", 1)[1].strip()
    if rhs.endswith(">"):
        rhs = rhs[:-1].strip()
    if not (rhs.startswith("[") and rhs.endswith("]")):
        return None
    inner = rhs[1:-1]
    if inner.startswith("not_"):
        return False
    return True


def nars_verdict_matches_env(stmt: Statement, frequency: float, core: str) -> Optional[bool]:
    """Map ONA's numeric judgment to a boolean and compare to the env's actual_truth."""
    pol = judgment_polarity_positive(core)
    if pol is None:
        return None
    pred_true = pol if frequency >= 0.5 else (not pol)
    return pred_true == stmt.actual_truth


def last_nars_state_for_core(
    lines: Sequence[str], core_norm: str
) -> Optional[Tuple[float, float, str]]:
    """Latest (frequency, confidence, line_kind) for this judgment core in the log so far."""
    for line in reversed(lines):
        p = parse_ona_line(line)
        if not p or p["kind"] not in ("Input", "Revised", "Selected"):
            continue
        if core_key_from_ona_body(str(p["body"])) == core_norm:
            return (float(p["frequency"]), float(p["confidence"]), str(p["kind"]))
    return None


def build_nars_dynamic_metrics(
    interaction_log: Sequence[Dict[str, object]],
    cumulative_lines: Sequence[str],
    facts: Sequence[Fact],
) -> Dict[str, object]:
    """
    Roll stepped rows into scalars and time series: per-agent NARS/env alignment,
    trust score vs designer map, final fact accuracy from cumulative ONA output, etc.
    """
    per_agent_scores: Dict[str, List[float]] = {}
    for row in interaction_log:
        m = row.get("nars_belief_matches_env_after")
        if m is None:
            continue
        src = str(row["source_agent"])
        per_agent_scores.setdefault(src, []).append(1.0 if m else 0.0)

    nars_learned_reliability = {
        a: sum(v) / len(v) for a, v in per_agent_scores.items() if v
    }

    true_map = _true_reliability_map()
    errs: List[float] = []
    for name, est in nars_learned_reliability.items():
        if name in true_map:
            errs.append(abs(est - true_map[name]))
    nars_trust_inference = 1.0 - (sum(errs) / len(errs)) if errs else 0.0

    known = [r for r in interaction_log if r.get("nars_belief_matches_env_after") is not None]
    nars_mean_belief_alignment = (
        sum(1 for r in known if r["nars_belief_matches_env_after"]) / len(known) if known else None
    )

    deceptive_hits = [
        r
        for r in interaction_log
        if str(r["source_agent"]).startswith("deceptive_")
        and r.get("nars_belief_matches_env_after") is not None
    ]
    nars_misinformation_resistance = (
        sum(1 for r in deceptive_hits if r["nars_belief_matches_env_after"]) / len(deceptive_hits)
        if deceptive_hits
        else None
    )

    deltas = [float(r["nars_confidence_delta"]) for r in interaction_log if r.get("nars_confidence_delta") is not None]
    nars_confidence_evolution_index = (
        sum(abs(d) for d in deltas) / len(deltas) if deltas else None
    )

    final_fact_hits = 0
    final_fact_n = 0
    for f in facts:
        st = Statement(0, "_env_", f.fact_id, f.proposition, f.truth, f.truth)
        ck = judgment_core_from_statement(st)
        snap = last_nars_state_for_core(cumulative_lines, ck)
        if snap is None:
            continue
        final_fact_n += 1
        if nars_verdict_matches_env(st, snap[0], ck):
            final_fact_hits += 1
    nars_belief_accuracy_final = (
        final_fact_hits / final_fact_n if final_fact_n else None
    )

    series: Dict[str, List[Tuple[int, float]]] = {}
    for row in interaction_log:
        conf = row.get("nars_confidence_after")
        if conf is None:
            continue
        prop = str(row["proposition"])
        ix = int(row["interaction_index"])
        series.setdefault(prop, []).append((ix, float(conf)))

    return {
        "nars_learned_reliability_by_agent": nars_learned_reliability,
        "nars_trust_inference_score": nars_trust_inference,
        "nars_mean_belief_alignment_per_step": nars_mean_belief_alignment,
        "nars_misinformation_resistance_after_deceptive": nars_misinformation_resistance,
        "nars_confidence_evolution_index": nars_confidence_evolution_index,
        "nars_belief_accuracy_final_vs_env": nars_belief_accuracy_final,
        "proposition_confidence_series": series,
        "interaction_steps_logged": len(interaction_log),
    }


# NARS-specific plots (built from stepped ONA log, not the Python tracker)


def visualize_nars_proposition_confidence_evolution(
    series: Dict[str, List[Tuple[int, float]]],
    save_path: Optional[str] = None,
    max_propositions: int = 8,
) -> None:
    """Line chart per proposition; y-axis zooms to the data so 0.85-0.95 isn't a flat line at the top."""
    if not _plotting_available():
        return
    ranked = sorted(series.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_propositions]
    if not ranked:
        return
    all_y: List[float] = []
    for _, pts in ranked:
        all_y.extend(float(y) for _, y in pts)
    y_lo = min(all_y)
    y_hi = max(all_y)
    span = y_hi - y_lo
    margin = 0.02 if span < 1e-9 else max(0.015, span * 0.12)
    ymin = max(0.0, y_lo - margin)
    ymax = min(1.05, y_hi + margin)
    if ymax - ymin < 0.06:
        mid = 0.5 * (ymin + ymax)
        ymin = max(0.0, mid - 0.03)
        ymax = min(1.05, mid + 0.03)

    plt.figure(figsize=(11, 6))
    for prop, pts in ranked:
        pts = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = prop if len(prop) < 48 else prop[:45] + "…"
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=label)
    plt.xlabel("Interaction index (each point = after one statement to NARS)")
    plt.ylabel("NARS confidence (last matching judgment)")
    plt.ylim(ymin, ymax)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    _finalize_plot(
        title="NARS: confidence evolution for propositions",
        xlabel="Interaction",
        ylabel="Confidence",
        save_path=save_path,
    )


def visualize_nars_learned_reliability_bar(
    learned: Dict[str, float], save_path: Optional[str] = None
) -> None:
    """Bar chart: fraction of post-step judgments where NARS matched env, broken down by agent."""
    if not _plotting_available():
        return
    if not learned:
        return
    names = sorted(learned.keys())
    vals = [learned[n] for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(max(8, len(names) * 0.35), 5))
    plt.bar(x, vals, color="teal", edgecolor="k", linewidth=0.3)
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("NARS alignment rate (post-step belief vs env)")
    plt.ylim(0, 1.05)
    _finalize_plot(
        title="NARS: learned source reliability (from belief alignment)",
        xlabel="Agent",
        ylabel="Rate",
        save_path=save_path,
    )


# OpenNARS / ONA subprocess bridge


class OpenNARSBridge:
    """
    Launches the shell jar, writes Narsese judgments to stdin, drains stdout on a worker thread.
    Stepped feeding is what we use for experiments so confidence can be read between inputs.
    """

    def __init__(self, launch_command: Sequence[str], working_dir: Optional[str] = None):
        self.launch_command = list(launch_command)
        self.working_dir = working_dir
        self.process: Optional[subprocess.Popen] = None
        self._stdout_queue: "queue.Queue[str]" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.process is not None:
            return

        if not self.launch_command:
            raise ValueError("launch_command cannot be empty.")

        self.process = subprocess.Popen(
            self.launch_command,
            cwd=self.working_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(
            target=self._read_stdout_loop,
            daemon=True
        )
        self._reader_thread.start()

        # Lab/GUI jars often quit when stdin isn't a TTY, or ignore stdin entirely.
        time.sleep(0.2)
        code = self.process.poll()
        if code is not None:
            tail = self._snapshot_stdout_lines(max_lines=25)
            raise RuntimeError(
                "NARS process exited right after launch "
                f"(exit {code}) for command {self.launch_command!r}. "
                f"Last output lines: {tail!r}"
            )

    def _read_stdout_loop(self) -> None:
        if self.process is None or self.process.stdout is None:
            return

        # Line-buffered text mode: blocks until a line arrives, which is fine on a daemon thread.
        for line in self.process.stdout:
            self._stdout_queue.put(line.rstrip("\n"))

    def _snapshot_stdout_lines(self, max_lines: int = 40) -> List[str]:
        lines: List[str] = []
        while len(lines) < max_lines:
            try:
                lines.append(self._stdout_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def send(self, text: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("OpenNARS process has not been started.")

        if not text.endswith("\n"):
            text += "\n"

        try:
            self.process.stdin.write(text)
            self.process.stdin.flush()
        except BrokenPipeError as err:
            code = self.process.poll()
            tail = self._snapshot_stdout_lines()
            raise RuntimeError(
                "Broken pipe: the Java process stopped reading stdin (crashed or exited). "
                f"exit={code}, stderr/stdout tail={tail!r}"
            ) from err

    def send_statement(self, statement: Statement, confidence: float = 0.90) -> None:
        """Judgment line only (no comment prefix)."""
        narsese_line = statement.to_narsese_judgment(confidence=confidence)
        self.send(narsese_line)

    def send_statements_stepped(
        self,
        statements: Sequence[Statement],
        confidence: float = 0.90,
        per_statement_timeout: float = 0.5,
        per_statement_max_lines: int = 900,
        pause_s: float = 0.02,
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        """
        One input at a time, then suck up stdout for a bounded window. That way each log row
        sees the belief state after that specific utterance (modulo how chatty ONA is).
        """
        cumulative: List[str] = []
        log_rows: List[Dict[str, object]] = []
        prev_conf: Dict[str, float] = {}
        for ix, statement in enumerate(statements):
            self.send_statement(statement, confidence=confidence)
            time.sleep(pause_s)
            chunk = self.read_available_output(
                max_lines=per_statement_max_lines,
                timeout=per_statement_timeout,
            )
            cumulative.extend(chunk)
            core = judgment_core_from_statement(statement)
            snap = last_nars_state_for_core(cumulative, core)
            prev_c = prev_conf.get(core)
            delta: Optional[float] = None
            if snap is not None and prev_c is not None:
                delta = float(snap[1]) - prev_c
            match_after: Optional[bool] = None
            if snap is not None:
                match_after = nars_verdict_matches_env(statement, float(snap[0]), core)
                prev_conf[core] = float(snap[1])
            log_rows.append(
                {
                    "interaction_index": ix,
                    "simulation_step": statement.step,
                    "source_agent": statement.source_agent,
                    "proposition": statement.proposition,
                    "fact_id": statement.fact_id,
                    "claimed_truth": statement.claimed_truth,
                    "actual_truth": statement.actual_truth,
                    "claim_matches_env": statement.claimed_truth == statement.actual_truth,
                    "judgment_core": core,
                    "nars_frequency_after": float(snap[0]) if snap else None,
                    "nars_confidence_after": float(snap[1]) if snap else None,
                    "nars_confidence_delta": delta,
                    "nars_belief_matches_env_after": match_after,
                }
            )
        return cumulative, log_rows

    def read_available_output(self, max_lines: int = 50, timeout: float = 0.2) -> List[str]:
        """Non-blocking-ish drain: grab whatever showed up before the timeout, up to max_lines."""
        lines: List[str] = []
        end_time = time.time() + timeout

        while len(lines) < max_lines and time.time() < end_time:
            try:
                remaining = max(0.0, end_time - time.time())
                line = self._stdout_queue.get(timeout=min(0.05, remaining or 0.01))
                lines.append(line)
            except queue.Empty:
                continue

        return lines

    def stop(self) -> None:
        if self.process is None:
            return

        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass

        try:
            self.process.terminate()
        except Exception:
            pass

        self.process = None


# Full experiment driver


def run_large_experiment(
    num_facts: int = 50,
    num_steps: int = 100,
    seed: Optional[int] = 42,
    nars_command: Optional[Sequence[str]] = None,
    working_dir: Optional[str] = None,
    nars_output_max_lines: int = 500,
    nars_output_timeout: float = 3.0,
    nars_stepped_statement_timeout: float = 0.45,
    nars_stepped_max_lines: int = 900,
) -> Tuple[List[Statement], BeliefTracker, Dict[str, object]]:
    """
    Build a long run, run Python metrics on it, optionally replay the same stream into ONA
    with stepped logging (see metrics keys nars_output, nars_interaction_log, nars_dynamic).
    """
    facts = generate_more_varied_facts(num_facts=num_facts, seed=seed)
    agents = build_large_agent_pool(seed=seed)

    sim = Simulation(facts=facts, agents=agents, seed=seed)
    statements = sim.run(num_steps=num_steps)

    tracker = BeliefTracker()
    for statement in statements:
        tracker.update(statement)

    metrics = run_all_metrics(
        statements=statements,
        belief_summary=tracker.belief_summary(),
        source_summary=tracker.source_summary(),
    )

    if nars_command:
        bridge = OpenNARSBridge(
            launch_command=nars_command,
            working_dir=working_dir
        )
        bridge.start()

        try:
            cum, log_rows = bridge.send_statements_stepped(
                statements,
                confidence=0.90,
                per_statement_timeout=nars_stepped_statement_timeout,
                per_statement_max_lines=max(nars_output_max_lines, nars_stepped_max_lines),
            )
            time.sleep(0.35)
            # Cheap extra drain: sometimes Revised lines land a beat after the per-input window.
            cum.extend(
                bridge.read_available_output(
                    max_lines=min(1200, max(nars_output_max_lines, 400) * 2),
                    timeout=nars_output_timeout,
                )
            )
            metrics["nars_output"] = cum
            metrics["nars_interaction_log"] = log_rows
            metrics["nars_dynamic"] = build_nars_dynamic_metrics(log_rows, cum, facts)
        finally:
            bridge.stop()

    return statements, tracker, metrics


# Entry point — tweak paths here for your machine


def _default_ona_install_dir() -> Path:
    """Where a typical clone of ONA sits if you put it next to this script."""
    return Path(__file__).resolve().parent / "OpenNARS-for-Applications-master"


def resolve_ona_home() -> Optional[str]:
    """
    Pick an OpenNARS-for-Applications root that contains the `NAR` launcher.

    Order: ONA_HOME, then OPENNARS_HOME, then a folder named OpenNARS-for-Applications-master
    alongside project.py. If none of those work, returns None so the rest of the script still runs
    without Java/ONA.
    """
    roots: List[Path] = []
    for key in ("ONA_HOME", "OPENNARS_HOME"):
        v = os.environ.get(key)
        if v:
            roots.append(Path(v).expanduser().resolve())
    roots.append(_default_ona_install_dir())
    for root in roots:
        nar = root / "NAR"
        if root.is_dir() and nar.exists():
            return str(root)
    return None


_ONA_HOME = resolve_ona_home()
# `NAR shell` is the stdin-friendly entry point for piping judgments.
NARS_COMMAND: Optional[List[str]] = (
    [f"{_ONA_HOME}/NAR", "shell"] if _ONA_HOME is not None else None
)

# Set ONA_PLOT_PREFIX to None to skip PNGs; filenames get that prefix in _plot_output_path.
ONA_PLOT_PREFIX: Optional[str] = "ona_experiment"
ONA_ENGINE_LOG_PATH: Optional[str] = "ona_engine_stdout.txt"


def main() -> None:
    # --- Short demo (no Java): prints stream + writes small_demo.nal ---
    facts = generate_environment_facts(num_facts=6, seed=42)
    agents: List[BaseAgent] = [
        TruthfulAgent("truthful_1"),
        DeceptiveAgent("deceptive_1"),
        NoisyAgent("noisy_1", seed=7),
        StrategicAgent("strategic_1", switch_interval=2),
    ]

    sim = Simulation(facts=facts, agents=agents, seed=99)
    statements = sim.run(num_steps=5)

    narsese_text = format_statements_as_narsese_document(statements, confidence=0.90)

    tracker = BeliefTracker()
    for statement in statements:
        tracker.update(statement)

    source_summary = tracker.source_summary()
    belief_summary = tracker.belief_summary()

    metrics = run_all_metrics(
        statements=statements,
        belief_summary=belief_summary,
        source_summary=source_summary
    )

    print("=== Statement Stream ===")
    for statement in statements:
        print(statement.to_text())

    print("\n=== Narsese Stream ===")
    print(narsese_text)

    print("\n=== Belief Summary ===")
    for row in belief_summary:
        print(row)

    print("\n=== Source Summary ===")
    for row in source_summary:
        print(row)

    print("\n=== Metrics ===")
    print("Belief accuracy:", metrics["belief_accuracy"])
    print("Trust inference score:", metrics["trust_inference_score"])
    print("Misinformation resistance:", metrics["misinformation_resistance"])

    save_narsese_experiment_file(statements, "small_demo.nal", confidence=0.90)

    # --- Heavier run: optionally pipes the same stream into ONA (needs `NAR` on disk or ONA_HOME). ---
    ona_statements, ona_tracker, ona_metrics = run_large_experiment(
        num_facts=10,
        num_steps=24,
        seed=42,
        nars_command=NARS_COMMAND,
        working_dir=_ONA_HOME,
        nars_output_max_lines=400,
        nars_output_timeout=4.0,
        nars_stepped_statement_timeout=0.5,
    )

    if NARS_COMMAND:
        print(f"\n=== ONA run — same {len(ona_statements)} judgments as above tracker ===")
    else:
        print(
            f"\n=== Large sim (Python only) — {len(ona_statements)} judgments ===\n"
            "ONA was skipped: no `NAR` launcher found. Your grader can set ONA_HOME to their checkout, "
            "or clone OpenNARS-for-Applications next to project.py under the default folder name."
        )
    print("=== Belief summary (Python tracker on that stream) ===")
    for row in ona_tracker.belief_summary():
        print(row)
    print("\n=== Source summary ===")
    for row in ona_tracker.source_summary():
        print(row)
    print("\n=== Metrics (Python) ===")
    print("Belief accuracy:", ona_metrics["belief_accuracy"])
    print("Trust inference score:", ona_metrics["trust_inference_score"])
    print("Misinformation resistance:", ona_metrics["misinformation_resistance"])

    nars_dyn = ona_metrics.get("nars_dynamic")
    if isinstance(nars_dyn, dict):
        print("\n=== NARS belief dynamics (after each judgment to ONA) ===")
        print("  interaction steps logged:", nars_dyn.get("interaction_steps_logged"))
        print("  NARS belief accuracy (final vs env):", nars_dyn.get("nars_belief_accuracy_final_vs_env"))
        print("  NARS trust inference (vs designer map):", nars_dyn.get("nars_trust_inference_score"))
        print("  NARS mean alignment per step:", nars_dyn.get("nars_mean_belief_alignment_per_step"))
        print("  NARS misinformation resistance (after deceptive):", nars_dyn.get("nars_misinformation_resistance_after_deceptive"))
        print("  NARS confidence evolution index:", nars_dyn.get("nars_confidence_evolution_index"))
        learned = nars_dyn.get("nars_learned_reliability_by_agent") or {}
        if learned:
            print("  NARS learned reliability by agent:", learned)

    inter_path = _plot_output_path("nars_interaction_log.json")
    with open(inter_path, "w", encoding="utf-8") as jf:
        json.dump(ona_metrics.get("nars_interaction_log", []), jf, indent=2)
    print(f"\nStepped interaction log (JSON): {inter_path}")

    nars_lines: List[str] = list(ona_metrics.get("nars_output", []))
    if nars_lines:
        if ONA_ENGINE_LOG_PATH:
            with open(ONA_ENGINE_LOG_PATH, "w", encoding="utf-8") as logf:
                logf.write("\n".join(nars_lines))
            print(f"\nONA stdout also saved to {ONA_ENGINE_LOG_PATH} ({len(nars_lines)} lines).")

        print("\n=== ONA engine stdout (first 40 lines; rest in log file) ===")
        for line in nars_lines[:40]:
            print(line)
        if len(nars_lines) > 40:
            print(f"... ({len(nars_lines) - 40} more lines in log file)")

    # Savefig path is absolute so running from another cwd still drops PNGs next to this file.
    if ONA_PLOT_PREFIX:
        rel_png = _plot_output_path(f"{ONA_PLOT_PREFIX}_reliability.png")
        all_png = _plot_output_path(f"{ONA_PLOT_PREFIX}_all_metrics.png")
    else:
        rel_png, all_png = None, None
    visualize_source_reliability_results(
        ona_metrics.get("source_reliability_results", []),
        save_path=rel_png,
    )
    visualize_all_metrics_image(ona_metrics, save_path=all_png)
    if ONA_PLOT_PREFIX and isinstance(nars_dyn, dict):
        evo_png = _plot_output_path(f"{ONA_PLOT_PREFIX}_nars_confidence_evolution.png")
        nars_rel_png = _plot_output_path(f"{ONA_PLOT_PREFIX}_nars_learned_reliability.png")
        series = nars_dyn.get("proposition_confidence_series")
        if isinstance(series, dict):
            visualize_nars_proposition_confidence_evolution(series, save_path=evo_png)
        learned_map = nars_dyn.get("nars_learned_reliability_by_agent")
        if isinstance(learned_map, dict):
            visualize_nars_learned_reliability_bar(learned_map, save_path=nars_rel_png)
        for p in (evo_png, nars_rel_png):
            if Path(p).is_file():
                print(f"\nWrote NARS plot ({Path(p).stat().st_size} bytes): {p}")
    if ONA_PLOT_PREFIX and rel_png and all_png:
        for p in (rel_png, all_png):
            # visualize_* returns without savefig if mpl is missing or there is nothing to draw.
            if Path(p).is_file():
                print(f"\nWrote plot ({Path(p).stat().st_size} bytes): {p}")

if __name__ == "__main__":
    main()