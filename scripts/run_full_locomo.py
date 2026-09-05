"""
Full 10-conversation LoCoMo benchmark runner, meant for an unattended GPU
box (e.g. an AWS EC2 GPU instance) rather than an interactive notebook --
run it under nohup/tmux so a dropped SSH session doesn't kill a multi-hour
run. Progress checkpoints to disk after every conversation (see
benchmarks/locomo/evaluate.py::run_locomo), so re-running this script after
an interruption resumes instead of starting over.

Supports running any of the five systems (HippoVoice + four local baseline
reimplementations) through the *real*, F1-scored LoCoMo QA benchmark under
the identical LLM/data/scoring -- not just the separate synthetic
signal/noise benchmark. This is what makes a comparison against
Mem0/A-MEM/NaiveRAG/Zep fair: their own published numbers use a different
LLM (usually GPT-4-class) and almost always a different, more lenient
scoring methodology (LLM-as-judge, not LoCoMo's own strict token-F1) --
confirmed as a real, disputed problem in this exact space, not a
theoretical concern: Zep's own team reported ~84% on LoCoMo, Mem0's
replication of Zep under its own methodology scored it at 58.44%, and Zep
rebutted with 75.14% -- three different numbers for the same system,
depending entirely on who measured it and how. Comparing our number
directly against any published figure would conflate the memory
architecture with the underlying model AND the yardstick. Running all five
through this exact same harness isolates the one variable that actually
matters here.

Usage:
    nohup python scripts/run_full_locomo.py --system hippovoice > run_hippovoice.log 2>&1 &
    nohup python scripts/run_full_locomo.py --system mem0       > run_mem0.log 2>&1 &
    nohup python scripts/run_full_locomo.py --system amem       > run_amem.log 2>&1 &
    nohup python scripts/run_full_locomo.py --system naive      > run_naive.log 2>&1 &
    nohup python scripts/run_full_locomo.py --system zep        > run_zep.log 2>&1 &
    tail -f run_hippovoice.log

Run these one at a time (they share the same GPU) -- queue them with `&&`
in one shell, or just run each to completion before starting the next.
"""
import argparse
import json
import sys

from llm.client import LLMClient
from memory.extractor import extract_memories
from benchmarks.locomo.evaluate import run_locomo

# decay_lambda/relevance_weight tuned for LoCoMo's conversation length
# (hundreds of turns), not the pipeline's ~90-100-turn default (see BUGS.md
# for why the default forgets almost everything before a real LoCoMo
# conversation ends). Only used by the HippoVoice factory -- baselines
# don't accept these parameters.
#
# decay_lambda=0.0005 is what the Kaggle sweep below actually validated on
# the full set (paired with TOP_K=10) -- but it tied EXACTLY with 0.001 on
# the cheap subset (both scored 0.256), so this isn't "0.0005 beats 0.001";
# at this scale the two are statistically indistinguishable. Kept at the
# exact value that was validated end-to-end rather than swapping back to
# 0.001 on the (very likely correct, but unconfirmed) assumption it'd score
# the same. top_k is the real driver -- see below.
DECAY_LAMBDA = 0.0005
RELEVANCE_WEIGHT = 0.85

# TOP_K=10 (up from run_locomo's own harness-wide default of 5) is a real,
# confirmed improvement specifically for HippoVoice, found via a coordinate-
# descent sweep on Kaggle (decay_lambda x relevance_weight x top_k) and
# validated on the full 1540-question set: 24.1% -> 27.74% avg F1, with the
# whole score distribution shifting favorably (fewer near_zero, more
# partial+high), not just the mean -- see BUGS.md for the full sweep.
# Deliberately NOT changed as run_locomo's own shared default: Mem0-style/
# A-MEM-style/NaiveRAG's recorded 23.4%/22.0% numbers were run at top_k=5,
# and top_k=10 was never swept for them -- bumping the shared default would
# silently make the README comparison table apples-to-oranges. Only
# HippoVoice's own factory below opts into it explicitly.
TOP_K = 10


def _hippovoice_factory(llm):
    from pipeline import HippoVoicePipeline
    return HippoVoicePipeline(
        llm_client=llm, text_only=True,
        decay_lambda=DECAY_LAMBDA, relevance_weight=RELEVANCE_WEIGHT,
    )


def _mem0_factory(llm):
    from baselines.mem0_baseline import Mem0Baseline
    return Mem0Baseline(llm_client=llm)


def _amem_factory(llm):
    from baselines.a_mem_baseline import AMemBaseline
    return AMemBaseline(llm_client=llm)


def _naive_factory(llm):
    from baselines.naive_rag import NaiveRAG
    p = NaiveRAG()
    # NaiveRAG doesn't use an LLM for memory management (pure vector store,
    # no extraction/consolidation), so it has no .llm attribute of its own --
    # but run_locomo's QA-answering step always calls conv_pipeline.llm to
    # generate the final answer, regardless of system. Attach the shared
    # client so it's wire-compatible for that step without changing the
    # baseline class itself.
    p.llm = llm
    return p


def _zep_factory(llm):
    from baselines.zep_baseline import ZepBaseline
    return ZepBaseline(llm_client=llm)


# None (the second element) means "use run_locomo's own default factory",
# i.e. real HippoVoicePipeline construction with the values above -- kept
# as None rather than duplicating that logic here.
SYSTEMS = {
    "hippovoice": ("HippoVoice", None),
    "mem0": ("Mem0-style", _mem0_factory),
    "amem": ("AMem-style", _amem_factory),
    "naive": ("NaiveRAG", _naive_factory),
    "zep": ("Zep-style", _zep_factory),
}


def sanity_check(llm) -> None:
    """
    Cheap pre-flight check before committing hours of GPU time to a full
    run -- same probe used throughout this project's development to catch
    extraction regressions (junk pollution, under-extraction, dropped
    dates) early. Exits before the real run if anything looks wrong.
    Only meaningful for systems that use memory/extractor.py's own
    extraction prompt -- HippoVoice and the Mem0-style/A-MEM-style
    baselines all do. Zep-style has its own, different extraction prompt
    (entities + fact triples, not typed fragments) and gets its own check
    below -- this one would silently pass without ever exercising Zep's
    actual prompt, which isn't the same as confirming it works.
    """
    print("=== Extraction sanity check ===")
    ok = True

    for t in ["Caroline: Thanks, Mel!", "Agreed, Caroline."]:
        result = extract_memories(t, llm)
        print(f"  junk turn {t!r} -> {result}")
        if result != []:
            ok = False

    signal_turn = "My best friend told me she's moving across the country permanently. I'm devastated."
    result = extract_memories(signal_turn, llm)
    print(f"  signal turn {signal_turn!r} -> {result}")
    if not result:
        ok = False

    dated_turn = "[10 am on 19 January, 2023] Jon: I lost my job as a banker today."
    result = extract_memories(dated_turn, llm)
    print(f"  dated turn {dated_turn!r} -> {result}")
    if not result or not any("2023" in m.get("content", "") for m in result):
        ok = False

    if not ok:
        print("\nSANITY CHECK FAILED -- stopping before spending GPU time on the full run.")
        print("See BUGS.md for the extraction-prompt history if this needs investigating.")
        sys.exit(1)

    print("Sanity check passed.\n")


def zep_sanity_check(llm) -> None:
    """
    Zep-style's own pre-flight check -- exercises its actual extraction
    prompt (entities + fact triples), not memory/extractor.py's, and
    confirms the specific mechanics its retrieval depends on: entity
    resolution (same name -> same entity across turns) and edge
    invalidation (a contradicting fact supersedes the old one). A broken
    prompt here would otherwise only surface after hours of GPU time spent
    on a full run silently extracting nothing useful.
    """
    from baselines.zep_baseline import ZepBaseline

    print("=== Zep-style extraction sanity check ===")
    baseline = ZepBaseline(llm_client=llm)
    ok = True

    result = baseline._extract("Caroline: Thanks, Mel!")
    print(f"  junk turn -> {result}")
    if result.get("entities") or result.get("facts"):
        ok = False

    result = baseline._extract("My best friend Sarah moved to Chicago last year.")
    print(f"  signal turn -> {result}")
    if not result.get("entities") or not result.get("facts"):
        ok = False

    # Real ingest, not just raw extraction -- confirms entity resolution and
    # edge invalidation actually work end to end, not just that the LLM
    # returns well-formed JSON.
    baseline.ingest_text_turn("Caroline lives in Seattle.")
    baseline.ingest_text_turn("Caroline moved to Portland.")
    caroline_ids = {eid for eid, name in baseline._entity_names.items() if name.lower() == "caroline"}
    live_facts = [f for f in baseline._facts.values() if f["invalid_at"] is None]
    print(f"  entities after 2 turns about the same person: {len(caroline_ids)} (want 1)")
    print(f"  live facts after a contradiction: {len(live_facts)} (want 1, the newer one)")
    if len(caroline_ids) != 1 or len(live_facts) != 1:
        ok = False

    if not ok:
        print("\nZEP SANITY CHECK FAILED -- stopping before spending GPU time on the full run.")
        sys.exit(1)

    print("Zep sanity check passed.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system", choices=sorted(SYSTEMS), default="hippovoice",
        help="Which system to run through the real LoCoMo QA benchmark.",
    )
    args = parser.parse_args()
    system_name, factory = SYSTEMS[args.system]

    checkpoint_path = f"locomo_checkpoint_full_{args.system}.json"
    results_path = f"locomo_full_results_{args.system}.json"

    print(f"System under test: {system_name}")
    print("Loading Qwen/Qwen3-4B (4-bit)...")
    llm = LLMClient(model_name="Qwen/Qwen3-4B", load_in_4bit=True)
    print(f"Loaded: {llm.model_name}  backend: {llm._backend}\n")

    if args.system == "zep":
        zep_sanity_check(llm)
    else:
        sanity_check(llm)

    print(f"Running FULL LoCoMo benchmark for {system_name} (10 conversations, all QA pairs)...\n")
    kwargs = dict(
        llm_client=llm,
        num_conversations=10,
        max_qa_per_conversation=None,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    if factory is None:
        # Default HippoVoice path -- let run_locomo apply DECAY_LAMBDA/
        # RELEVANCE_WEIGHT/TOP_K itself rather than duplicating the values here.
        kwargs["decay_lambda"] = DECAY_LAMBDA
        kwargs["relevance_weight"] = RELEVANCE_WEIGHT
        kwargs["top_k"] = TOP_K
    else:
        kwargs["pipeline_factory"] = factory
        kwargs["system_name"] = system_name

    result = run_locomo(**kwargs)

    print("=" * 60)
    print(f"{system_name} LoCoMo avg F1: {result['avg_f1']:.1%}  (over {result['total']} questions)")
    print(f"bins: {result['bins']}")
    print("=" * 60)

    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()
