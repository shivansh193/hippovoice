"""
Full 10-conversation LoCoMo benchmark runner, meant for an unattended GPU
box (e.g. an AWS EC2 GPU instance) rather than an interactive notebook --
run it under nohup/tmux so a dropped SSH session doesn't kill a multi-hour
run. Progress checkpoints to disk after every conversation (see
benchmarks/locomo/evaluate.py::run_locomo), so re-running this script after
an interruption resumes instead of starting over.

Usage:
    nohup python scripts/run_full_locomo.py > run_full_locomo.log 2>&1 &
    tail -f run_full_locomo.log
"""
import json
import sys

from llm.client import LLMClient
from memory.extractor import extract_memories
from benchmarks.locomo.evaluate import run_locomo

CHECKPOINT_PATH = "locomo_checkpoint_full.json"
RESULTS_PATH = "locomo_full_results.json"

# Same decay/relevance values validated on Kaggle this session -- tuned for
# LoCoMo's conversation length (hundreds of turns), not the pipeline's
# ~90-100-turn default (see BUGS.md for why the default forgets almost
# everything before a real LoCoMo conversation ends).
DECAY_LAMBDA = 0.001
RELEVANCE_WEIGHT = 0.85


def sanity_check(llm) -> None:
    """
    Cheap pre-flight check before committing hours of GPU time to a full
    run -- same probe used throughout this project's development to catch
    extraction regressions (junk pollution, under-extraction, dropped
    dates) early. Exits before the real run if anything looks wrong.
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


def main() -> None:
    print("Loading Qwen/Qwen3-4B (4-bit)...")
    llm = LLMClient(model_name="Qwen/Qwen3-4B", load_in_4bit=True)
    print(f"Loaded: {llm.model_name}  backend: {llm._backend}\n")

    sanity_check(llm)

    print("Running FULL LoCoMo benchmark (10 conversations, all QA pairs)...\n")
    result = run_locomo(
        llm_client=llm,
        num_conversations=10,
        max_qa_per_conversation=None,
        checkpoint_path=CHECKPOINT_PATH,
        decay_lambda=DECAY_LAMBDA,
        relevance_weight=RELEVANCE_WEIGHT,
        verbose=True,
    )

    print("=" * 60)
    print(f"LoCoMo avg F1: {result['avg_f1']:.1%}  (over {result['total']} questions)")
    print(f"bins: {result['bins']}")
    print("=" * 60)

    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
