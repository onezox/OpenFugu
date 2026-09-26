# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: Learning to Orchestrate Agents in NL with the Conductor (arXiv:2512.04388, Sakana AI). Independent reimplementation of the workflow-DAG executor from the paper; no Sakana source code is copied.
"""
ultra.py — Fugu-Ultra's Conductor line: instead of routing one worker per turn
(that's mini.py / TRINITY), a Conductor model emits an ENTIRE agentic workflow in
one shot — three equal-length lists (model_id / subtasks / access_list) forming
a DAG over the worker pool — which is validated and then executed in order.

Provenance, stated honestly:
  [EXEC]  the execution engine — 3-list parse, DAG order, access-list visibility
          injection — follows the TRINITY/Conductor authors' conductor_engine.py
          + conductor_utils.py. Parsing and validation are stricter than theirs:
          any rule violation raises WorkflowError before a worker is called.
  [DOC]   the Conductor is any model reached through litellm (for example a
          fine-tuned one behind an OpenAI-compatible API), configured with the
          FUGU_CONDUCTOR_* environment variables.

Usage:
  FUGU_CONDUCTOR_MODEL=... FUGU_API_KEY=... \
  python -m openfugu.ultra --query "..." --slot-models <csv of worker model ids>
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from openfugu.llm import (ConfigError, Endpoint, LLMCallError, WorkerPool, check_endpoint,
                          complete, conductor_endpoint, configure_logging, parse_slot_models)

logger = logging.getLogger("openfugu.ultra")

MAX_STEPS = 5                      # [DOC] Conductor workflows up to 5 steps
CONDUCTOR_ATTEMPTS = 3
CONDUCTOR_TIMEOUT_S = 120.0
CONDUCTOR_MAX_TOKENS = 2048
CONDUCTOR_TEMPERATURE = 0.2

# Accepted spellings of each list label; the first one is canonical.
LABELS = {
    "model_id": ("model_id", "model_ids", "model id", "model ids"),
    "subtasks": ("subtasks", "subtask"),
    "access_list": ("access_list", "access list", "access"),
}
# A label counts only at the start of a line and only when "[" follows the
# ":" or "=" separator (whitespace aside), so labels in reasoning are ignored.
_LABEL_PATTERNS = {
    name: re.compile(r"^[ \t]*(?:" + "|".join(re.escape(a) for a in aliases)
                     + r")[ \t]*[:=]\s*(?=\[)", re.I | re.M)
    for name, aliases in LABELS.items()
}
_SMART_QUOTES = str.maketrans("“”‘’", "\"\"''")
ACCESS_RULE = 'must be [], "all", ["all"] or a list of indices of earlier steps'

_NO_VALUE = object()          # WorkflowError without an offending value
_NOT_A_LITERAL = object()     # _literal() could not parse its input


class WorkflowError(ValueError):
    """The Conductor's output breaks a workflow rule.

    `field` is the list ("model_id", "subtasks", "access_list") or "workflow"
    for whole-workflow rules, `index` the step (or None), `value` the offending
    value, and `rule` the rule it breaks.
    """

    def __init__(self, field: str, index: int | None, value: object, rule: str) -> None:
        self.field, self.index, self.value, self.rule = field, index, value, rule
        where = field if index is None else f"{field}[{index}]"
        if value is _NO_VALUE:
            super().__init__(f"{where}: {rule}")
        else:
            shown = repr(value)
            if len(shown) > 80:
                shown = shown[:77] + "..."
            super().__init__(f"{where} = {shown}: {rule}")


# ---- 3-list parsing (after conductor_utils._extract_any) [EXEC] ---------------
def _balanced_list(text: str, start: int) -> str | None:
    """The bracketed list opening at text[start], or None if it never closes.
    Brackets inside quoted strings do not count."""
    depth, quote, escaped = 0, None, False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _literal(raw: str) -> object:
    """Parse a Python literal, else JSON; `_NOT_A_LITERAL` if neither works."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _NOT_A_LITERAL


def _extract_list(text: str, name: str) -> list:
    """The list after the LAST line-leading `name` label in `text`."""
    matches = list(_LABEL_PATTERNS[name].finditer(text))
    if not matches:
        raise WorkflowError(name, None, _NO_VALUE,
                            f"missing: no line starts with '{name}:' followed by a list")
    start = matches[-1].end()
    # Curly quotes are a fallback: normalising them first would corrupt curly
    # quotes that sit inside properly quoted strings.
    candidates = [text]
    normalised = text.translate(_SMART_QUOTES)          # 1:1, so `start` stays valid
    if normalised != text:
        candidates.append(normalised)
    for candidate in candidates:
        raw = _balanced_list(candidate, start)
        if raw is not None:
            value = _literal(raw)
            if value is not _NOT_A_LITERAL:
                return value
    raise WorkflowError(name, None, text[start:start + 60],
                        "is not a complete Python or JSON list literal")


def parse_workflow(text: str) -> tuple[list, list, list]:
    """Extract the raw (model_id, subtasks, access_list) lists from Conductor output.

    Each label must start a line and be followed by ':' or '=' and the list.
    When a label appears more than once, the last occurrence wins. Raises
    WorkflowError if a list is missing or is not a Python/JSON literal.
    """
    return (_extract_list(text, "model_id"), _extract_list(text, "subtasks"),
            _extract_list(text, "access_list"))


# ---- validation ------------------------------------------------------------
@dataclass(frozen=True)
class Workflow:
    """A validated workflow. `sees[t]` lists the earlier steps step t can see."""

    model_ids: list[int]
    subtasks: list[str]
    access_list: list
    sees: list[list[int]]


def _is_all(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() == "all"


def _resolve_access(entry: object, step: int) -> list[int]:
    """Earlier steps visible to `step` (choose_position semantics). [EXEC]"""
    if _is_all(entry) or (isinstance(entry, list) and len(entry) == 1 and _is_all(entry[0])):
        return list(range(step))
    if not isinstance(entry, list):
        raise WorkflowError("access_list", step, entry, ACCESS_RULE)
    for ref in entry:
        if isinstance(ref, bool) or not isinstance(ref, int):
            raise WorkflowError("access_list", step, entry, ACCESS_RULE)
        if not 0 <= ref < step:
            earlier = {0: "none", 1: "step 0"}.get(step, f"steps 0 to {step - 1}")
            raise WorkflowError("access_list", step, entry,
                                f"step {step} can only see earlier steps ({earlier}), "
                                f"not step {ref}")
    return sorted(set(entry))


def validate_workflow(model_ids: list, subtasks: list, access_list: list,
                      pool_size: int) -> Workflow:
    """Check every workflow rule and resolve visibility; raise WorkflowError on the first break.

    Rules: all three lists have the same length, 1 to MAX_STEPS steps; each
    model_id is an integer worker index in [0, pool_size); each subtask is a
    non-empty string; each access_list entry is [], "all", ["all"] (any case)
    or a list of indices of earlier steps.
    """
    lengths = (len(model_ids), len(subtasks), len(access_list))
    if len(set(lengths)) != 1:
        raise WorkflowError("workflow", None, _NO_VALUE,
                            f"model_id, subtasks and access_list have {lengths[0]}, "
                            f"{lengths[1]} and {lengths[2]} entries; all three lists "
                            f"must have the same length")
    if not 1 <= lengths[0] <= MAX_STEPS:
        raise WorkflowError("workflow", None, _NO_VALUE,
                            f"has {lengths[0]} steps; a workflow must have 1 to "
                            f"{MAX_STEPS} steps")
    for i, model_id in enumerate(model_ids):
        if isinstance(model_id, bool) or not isinstance(model_id, int):
            raise WorkflowError("model_id", i, model_id, "must be an integer worker index")
        if not 0 <= model_id < pool_size:
            raise WorkflowError("model_id", i, model_id,
                                f"must be a worker index from 0 to {pool_size - 1}")
    for i, subtask in enumerate(subtasks):
        if not isinstance(subtask, str) or not subtask.strip():
            raise WorkflowError("subtasks", i, subtask, "must be a non-empty string")
    sees = [_resolve_access(entry, step) for step, entry in enumerate(access_list)]
    return Workflow(list(model_ids), [s.strip() for s in subtasks], list(access_list), sees)


# ---- the Conductor prompt ----------------------------------------------------
def conductor_prompt(query: str, pool: Sequence[str]) -> list[dict]:
    """The exact [system, user] messages the Conductor model receives."""
    workers = "\n".join(f"  {i}: {name}" for i, name in enumerate(pool))
    system = (
        "You are a Conductor that orchestrates a pool of worker LLMs to solve a task. "
        "Design an agentic workflow as THREE equal-length Python lists:\n"
        "  model_id   = [int, ...]   # which worker (0-indexed) runs each step\n"
        "  subtasks   = [str, ...]   # the natural-language instruction for each step\n"
        "  access_list= [list, ...]  # for each step, the indices of EARLIER steps whose\n"
        "                            # outputs that step may see ([] = none, \"all\" = every earlier step)\n"
        "Rules: lists must be equal length (<=5 steps); every model_id must be one of the "
        "worker indices listed below; access_list may only reference strictly earlier steps "
        "(it is a DAG executed in order); workers never see the user question, only their "
        "own subtask and the outputs they may access, so every subtask must be "
        "self-contained; the LAST step's output is the final answer. Pick workers to match "
        "each subtask's demands.\n\n"
        f"AVAILABLE LANGUAGE MODELS:\n{workers}\n\n"
        "Output the three lists explicitly as 'model_id: [...]', 'subtasks: [...]', "
        "'access_list: [...]', each on its own line. You may reason first, but the three "
        "lists must appear."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"USER QUESTION: {query}"}]


def conduct(endpoint: Endpoint, query: str, pool: Sequence[str]) -> str:
    """Ask the Conductor model for a workflow; returns its raw completion."""
    return complete(endpoint, conductor_prompt(query, pool), max_tokens=CONDUCTOR_MAX_TOKENS,
                    temperature=CONDUCTOR_TEMPERATURE, timeout=CONDUCTOR_TIMEOUT_S,
                    attempts=CONDUCTOR_ATTEMPTS)


# ---- execution ---------------------------------------------------------------
WorkerFn = Callable[[str, list, int], str]   # (subtask, messages, agent_id) -> reply


@dataclass
class Step:
    """One executed step: its worker, subtask, visible steps and reply."""

    idx: int
    agent_id: int
    subtask: str
    sees: list[int]
    reply: str


@dataclass
class UltraResult:
    """The executed workflow; `final` is the last step's output."""

    final: str
    workflow: Workflow
    steps: list[Step] = field(default_factory=list)


class ConductorExecutor:
    """Execute a validated workflow over the worker pool, in step order. [EXEC]

    Each step prompts its worker with its subtask plus the outputs of the steps
    it may see, injected as <Agent N response> blocks (the engine's exact
    context-assembly format)."""

    def __init__(self, worker: WorkerFn) -> None:
        self.worker = worker

    def execute(self, workflow: Workflow) -> UltraResult:
        """Run every step and return the result; the last step's output is final."""
        res = UltraResult(final="", workflow=workflow)
        outputs: list[str] = []
        for t, (mid, sub, sees) in enumerate(zip(workflow.model_ids, workflow.subtasks,
                                                 workflow.sees)):
            ctx = "".join(
                f"\n<Subtask assigned to Agent {workflow.model_ids[j]}>{workflow.subtasks[j]}"
                f"</Subtask assigned to Agent {workflow.model_ids[j]}>"
                f"\n<Agent {workflow.model_ids[j]} response>{outputs[j].strip()}"
                f"</Agent {workflow.model_ids[j]} response>"
                for j in sees)
            user = (f"USER QUESTION context:\n{ctx}\n\nYour subtask: {sub}"
                    if ctx else f"Your subtask: {sub}")
            reply = self.worker(sub, [{"role": "user", "content": user}], mid)
            outputs.append(reply)
            res.steps.append(Step(t, mid, sub, sees, reply))
            logger.info("step %d: agent=%d sees=%s reply_chars=%d", t, mid, sees, len(reply))
        res.final = outputs[-1]                             # last step = answer [EXEC]
        return res


# ---- CLI -----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run one query: the Conductor designs a workflow, the pool executes it."""
    ap = argparse.ArgumentParser(description="Fugu-Ultra: a Conductor designs a workflow, "
                                             "the worker pool executes it.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--slot-models", metavar="CSV", required=True,
                    help="comma-separated litellm worker model ids; worker i is entry i")
    args = ap.parse_args(argv)
    configure_logging(logging.WARNING)

    try:
        pool = parse_slot_models(args.slot_models)
        workers = WorkerPool.from_env(pool)
        workers.check()
        endpoint = conductor_endpoint()
        check_endpoint(endpoint, "Conductor")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1
    print(f"conductor: {endpoint.model}")
    print(f"workers: {len(pool)} slots")
    print(f"query: {args.query}\n")

    try:
        completion = conduct(endpoint, args.query, pool)
    except LLMCallError as exc:
        print(f"Conductor call failed: {exc}", file=sys.stderr)
        return 1
    try:
        workflow = validate_workflow(*parse_workflow(completion), pool_size=len(pool))
    except WorkflowError as exc:
        print(f"invalid workflow: {exc}\n\nConductor output:\n{completion[:800]}",
              file=sys.stderr)
        return 1

    print(f"workflow: model_id={workflow.model_ids}  access_list={workflow.access_list}"
          f"  ({len(workflow.subtasks)} steps)\n")
    try:
        res = ConductorExecutor(workers).execute(workflow)
    except LLMCallError as exc:
        print(f"worker call failed: {exc}", file=sys.stderr)
        return 1
    for step in res.steps:
        print(f"  step {step.idx}: agent={step.agent_id} ({pool[step.agent_id]}) "
              f"sees={step.sees}")
        print(f"    subtask: {step.subtask[:80]}")
        print(f"    -> {step.reply.strip()[:90]}")
    print(f"\nfinal answer (step {len(res.steps) - 1} output):\n{res.final.strip()[:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
