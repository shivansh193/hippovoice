"""
EasyEditWeightEditor -- the real WeightEditor implementation (see
baselines/weight_edit_baseline.py for the interface and the mock that
validates everything except this file). Wraps EasyEdit's ROME/MEMIT on
GPT-2 XL. Needs a GPU; not exercised by the local test suite at all --
validated on Kaggle instead (see kaggle_full_benchmark.ipynb).

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

        # Confirmed on a real run: the shipped hparams/{ROME,MEMIT}/gpt2-xl.yaml
        # hardcodes model_name: "./hugging_cache/gpt2-xl" -- a local-directory
        # convention for someone who already downloaded the model by hand, not
        # a HuggingFace Hub repo id. Passed straight through, this fails with
        # `HFValidationError: Repo id must be in the form 'repo_name' or
        # 'namespace/repo_name'` before ever touching a real model. Override
        # it with the actual public Hub id so from_pretrained downloads it
        # like any other model reference in this project.
        hparams.model_name = "gpt2-xl"
        self.editor = BaseEditor.from_hparams(hparams)

        # Confirmed as a real OOM on a Kaggle T4 (14MB short of the card's
        # ~14.5GB, right at the edit itself) and confirmed from EasyEdit's
        # own compute_v.py source: ROME's edit step runs ONE forward+backward
        # pass through all 48 layers, but batches ~15-20 reworded versions of
        # the same sentence together in that single pass (context_templates,
        # used so the edit generalizes across phrasings) -- that's what
        # actually blows the memory budget, not the ~6GB of plain weights.
        # A forward+backward pass needs to keep every layer's intermediate
        # activations in memory so backward can use them; with ~16 sentences
        # batched through 48 layers, that adds up fast.
        # gradient_checkpointing_enable() is the standard fix for this
        # specific shape of problem: instead of storing every layer's
        # activations, it throws most of them away during forward and
        # recomputes them on demand during backward. Same math, same
        # result -- IF nothing else in the model behaves differently
        # between the two passes. EasyEdit's compute_v.py never enables
        # this itself; it just calls forward on whatever model object it's
        # handed, so turning it on here (our own wrapper) is enough -- no
        # EasyEdit code needs touching.
        self.editor.model.gradient_checkpointing_enable()

        # Confirmed as a real, silent correctness bug, not a guess:
        # HuggingFace gates checkpointing with
        # `if self.gradient_checkpointing and self.training`, so making it
        # actually engage requires calling model.train() during the edit
        # (see edit() below) -- but GPT-2 XL has real dropout (~10%) on
        # attention/residuals/embeddings, and train() mode turns that back
        # on. A live Kaggle run showed exactly the failure this risks: the
        # OOM was gone, but "AFTER edit" came back byte-identical to
        # "BEFORE edit" -- the edit silently computed to ~nothing, because
        # ROME's correction vector depends on precise activations, and
        # dropout noise during that computation corrupted it. This is
        # worse than a crash: it "succeeds" with a wrong answer, exactly
        # the failure mode the sanity check two cells down exists to catch
        # (and did catch, here). Zeroing every Dropout module's p makes
        # dropout a true no-op in either train() or eval() mode, so
        # train()-for-checkpointing during edit() is safe: forward is
        # numerically identical to eval() mode, just recomputed instead of
        # stored during backward. GPT-2 has no BatchNorm (only LayerNorm,
        # which has no train/eval distinction), so this is the only
        # train()-vs-eval() behavioral difference that mattered here.
        import torch
        for module in self.editor.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

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
        #
        # model.train() here is confirmed necessary, not decorative: a real
        # Kaggle run showed load()'s gradient_checkpointing_enable() call
        # barely reduced peak memory (a T4 OOM's "free" memory only moved
        # from 6.8MB to 10.8MB, still short) -- because HuggingFace's own
        # transformer blocks gate checkpointing with
        # `if self.gradient_checkpointing and self.training`, and
        # BaseEditor.from_hparams() leaves the model in eval() mode.
        # Checkpointing was enabled but never actually engaging. Safe to
        # flip to train() here specifically because load() already zeroed
        # every Dropout module's p -- otherwise this exact switch silently
        # broke the edit itself on the very next real run (see load()'s
        # comment: "AFTER edit" came back identical to "BEFORE edit").
        # Wrapped in try/finally so eval() mode is restored even if the
        # edit raises -- generate() also defensively re-forces it below,
        # since generation should never run under an unexpected mode
        # regardless of how it got there.
        self.editor.model.train()
        try:
            self.editor.edit(
                prompts=[prompt],
                target_new=[target_new],
                subject=[subject],
                keep_original_weight=False,
                verbose=False,
            )
        finally:
            self.editor.model.eval()

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        tok = self.editor.tok
        model = self.editor.model
        model.eval()  # defensive -- generation must never see train()-mode dropout
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
