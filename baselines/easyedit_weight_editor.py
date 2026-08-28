"""
EasyEditWeightEditor -- the real WeightEditor implementation (see
baselines/weight_edit_baseline.py for the interface and the mock that
validates everything except this file). Wraps EasyEdit's ROME/MEMIT on
GPT-2 XL. Needs a GPU; not exercised by the local test suite at all --
validated on Kaggle instead (see kaggle_weight_edit_benchmark.ipynb).

Kept as its own module, not added to weight_edit_baseline.py, specifically
so importing that module (which the local mock-based tests do) never
requires EasyEdit or torch/CUDA to be installed. Only this file, and only
when actually instantiated, needs the real EasyEdit repo cloned alongside
this project.

Everything below is either directly confirmed against EasyEdit's own
source/hparams files (not guessed) or flagged as a specific inference this
project hasn't validated with a real run yet -- see the comments at each
such point. Two things in particular need a cheap single-edit sanity check
on Kaggle before trusting a full LoCoMo run, exactly like the extraction-
prompt saga in BUGS.md:
  1. That `keep_original_weight=False` per edit() call is really what
     makes edits accumulate across successive calls within one
     conversation (rather than each call silently restoring itself) --
     confirmed from EasyEdit's source that ROME/MEMIT mutate
     `editor.model` in place, but the exact effect of this flag across
     *repeated* single-edit calls (our per-turn interface, not EasyEdit's
     own batched-edit examples) hasn't been run for real yet.
  2. That the full-state-dict snapshot/reload in reset() actually restores
     generation behavior to pristine, not just parameter values that
     happen to look identical.

Method defaults to ROME, not MEMIT: confirmed by reading both hparams
files directly (hparams/ROME/gpt2-xl.yaml has `mom2_adjustment: false`;
hparams/MEMIT/gpt2-xl.yaml has `mom2_adjustment: true` with
`mom2_n_samples: 100000`) -- MEMIT needs precomputed/computed covariance
statistics over 100k wikipedia samples before its first edit, a real,
unmeasured cost this project hasn't paid yet. ROME needs none of that.
Get ROME validated and running first; MEMIT is a same-interface swap
(`method="MEMIT"`) once it is.
"""

import copy


class EasyEditWeightEditor:
    def __init__(self, easyedit_dir: str, method: str = "ROME"):
        """
        easyedit_dir: path to a git-cloned EasyEdit repo (pip alone doesn't
        ship the hparams/*.yaml files ROMEHyperParams.from_hparams() reads,
        so a clone is required regardless -- see kaggle_weight_edit_benchmark.ipynb).
        method: "ROME" or "MEMIT", matching a hparams/<method>/gpt2-xl.yaml
        under easyedit_dir.
        """
        self.easyedit_dir = easyedit_dir
        self.method = method
        self.editor = None
        self._original_state_dict = None

    def load(self):
        import sys

        if self.easyedit_dir not in sys.path:
            sys.path.insert(0, self.easyedit_dir)

        # Imported lazily, inside load(), specifically so this module can
        # be imported (e.g. by anything that just wants the interface or
        # is running the local mock-based test suite) without EasyEdit or
        # torch/CUDA installed at all -- only actually calling load() pays
        # that cost.
        from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams

        hparams_cls = ROMEHyperParams if self.method == "ROME" else MEMITHyperParams
        # Confirmed against EasyEdit's own edit.py test_ROME(): from_hparams
        # takes the path WITHOUT a .yaml extension (it appends that itself).
        hparams_path = f"{self.easyedit_dir}/hparams/{self.method}/gpt2-xl"
        hparams = hparams_cls.from_hparams(hparams_path)
        self.editor = BaseEditor.from_hparams(hparams)

        # Full CPU snapshot of pristine weights, taken once right after
        # load. reset() reloads this rather than relying on EasyEdit's own
        # per-edit weights_copy/restore path, because that mechanism only
        # ever covers the single most recent edit's touched parameters --
        # it has no way to roll back however many edits accumulated over
        # an entire conversation. This is deliberately independent of
        # whatever keep_original_weight actually does internally.
        self._original_state_dict = copy.deepcopy(self.editor.model.state_dict())

    def edit(self, prompt: str, subject: str, target_new: str) -> None:
        # keep_original_weight=False: confirmed from EasyEdit's source that
        # ROME/MEMIT mutate editor.model's weights in place -- we want each
        # successive edit() call in a conversation to stack on top of the
        # previous ones (sequential editing is the entire point of this
        # benchmark, see weight_edit_baseline.py's module docstring), not
        # get quietly undone. reset() is the only place weights should
        # actually roll back.
        self.editor.edit(
            prompts=[prompt],
            target_new=[target_new],
            subject=[subject],
            keep_original_weight=False,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        tok = self.editor.tok
        model = self.editor.model
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tok.eos_token_id,
        )
        full_text = tok.decode(output_ids[0], skip_special_tokens=True)
        # generate() on a causal LM echoes the prompt back at the start of
        # its output -- strip it so callers (WeightEditBaseline's QA step)
        # get just the completion, matching MockWeightEditor's contract.
        return full_text[len(prompt):].strip()

    def reset(self) -> None:
        if self._original_state_dict is not None:
            self.editor.model.load_state_dict(self._original_state_dict)
