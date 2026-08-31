"""Krea 2 — instruction image-editing (in-context) extension.

Additive subclass of Krea2Model that turns the T2I model into an editor with
**dual conditioning** (the Qwen-Image-Edit recipe), reusing Krea's components:

  * semantic path  — the source image is fed into the Qwen3-VL text encoder so the
    instruction is image-grounded; image-patch hidden states are removed before
    the VLM result becomes DiT context (``encode_control_in_text_embeddings``);
  * appearance path — the source latent is patchified and concatenated as a block
    of **clean** tokens before the (noisy) target tokens in the single-stream
    sequence, distinguished only by RoPE **frame axis = 1** (h,w stay aligned).

The T2I path in ``krea2.py`` is untouched. arch: ``krea2_edit``.
"""
import os
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, List, Optional, Sequence
from einops import rearrange, repeat

from toolkit.config_modules import ModelConfig
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds

from .krea2 import Krea2Model
from .src.mmdit import SingleStreamDiT
from .src.pipeline import pad_text_features, predict_velocity
from .src.text_encoder import (
    SELECT_LAYERS,
    PROMPT_TEMPLATE_ENCODE_PREFIX,
    PROMPT_TEMPLATE_ENCODE_SUFFIX,
    PROMPT_TEMPLATE_ENCODE_START_IDX,
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


def krea2_edit_ragged_collation(batch):
    """Keep raw edit references as a per-sample list during ai-toolkit collation.

    ai-toolkit's stock ``DataLoaderBatchDTO`` concatenates ``control_tensor``. That
    is correct for ControlNet-like inputs, but it loses the independent geometry
    contract of this model (and fails outright when two source images have different
    H/W).  The DTO already has a ragged ``control_tensor_list`` field, so normalize
    *every* raw edit control into that representation immediately before it builds
    the batch.  Targets remain normal bucketed tensors.

    This is deliberately a module-level function: PyTorch can pickle it as the
    DataLoader ``collate_fn`` on spawn platforms, so workers import this extension
    and run the same conversion.
    """
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

    # Bucketed ai-toolkit datasets yield a list of FileItemDTOs directly; the
    # non-bucketed DataLoader passes the same list as its batch.  Do not recurse
    # into tensors: a control image itself is a Tensor, not a batch item.
    if not isinstance(batch, (list, tuple)):
        batch = [batch]
    for item in batch:
        ctrl = getattr(item, "control_tensor", None)
        if ctrl is None:
            continue
        refs = getattr(item, "control_tensor_list", None)
        if refs is None:
            refs = [ctrl]
        else:
            # This should not normally occur (ai-toolkit uses one field or the
            # other), but preserve both sources if a third-party loader set both.
            refs = list(refs) + [ctrl]
        item.control_tensor = None
        item.control_tensor_list = refs
    return DataLoaderBatchDTO(file_items=list(batch))


def install_ragged_control_collator():
    """Install the edit-only collator before ai-toolkit constructs its DataLoader."""
    import toolkit.data_loader as data_loader

    if getattr(data_loader, "_krea2_edit_ragged_controls", False):
        return
    data_loader.dto_collation = krea2_edit_ragged_collation
    data_loader._krea2_edit_ragged_controls = True
    print("[krea2_edit] ragged raw-control collator ENABLED "
          "(references stay per-sample; targets stay bucketed)")


# Grounding (semantic path) resolution policy. These defaults MATCH the inference
# node: comfyui-krea2edit's Krea2EditTextEncode defaults `grounding_px` to 768, and
# the released krea2-identity-edit LoRAs trained with per-step jitter down to 384.
# Env vars still override at launch (0 disables the cap = native-resolution
# grounding, which is NOT what the released weights were trained with).
GROUNDING_MAX_PX_DEFAULT = "768"
GROUNDING_JITTER_MIN_DEFAULT = "384"


def grounding_settings():
    """(max_px, jitter_min) for the Qwen3-VL grounding image; env overrides defaults."""
    return (
        int(os.environ.get("GROUNDING_MAX_PX", GROUNDING_MAX_PX_DEFAULT)),
        int(os.environ.get("GROUNDING_JITTER_MIN", GROUNDING_JITTER_MIN_DEFAULT)),
    )


def _img_tokens_and_pos(latent: torch.Tensor, patch: int, frame: int):
    """Patchify a (B,C,h,w) latent → (tokens (B,L,C*p*p), pos (B,L,3), mask (B,L)).

    pos axes are (frame, h, w); ``frame`` distinguishes source (1) from target (0)
    while h,w stay aligned for preservation.
    """
    b, _, h, w = latent.shape
    h_, w_ = h // patch, w // patch
    ids = torch.zeros((h_, w_, 3), device=latent.device)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=latent.device)[:, None]
    ids[..., 2] = torch.arange(w_, device=latent.device)[None, :]
    pos = repeat(ids, "h w t -> b (h w) t", b=b, t=3)
    mask = torch.ones(b, h_ * w_, device=latent.device, dtype=torch.bool)
    tok = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    return tok, pos, mask


def predict_velocity_edit(
    model: SingleStreamDiT,
    target_latents: torch.Tensor,   # (B, C, h, w) noisy
    source_latents,                 # (B, C, h, w) clean reference, OR list of them (multi-ref)
    t: torch.Tensor,
    context: torch.Tensor,          # (B, Lt, n*d)
    text_mask: torch.Tensor,        # (B, Lt)
) -> torch.Tensor:
    """Sequence: [text | ref_1(frame=1) | ref_2(frame=2) | ... | target(frame=0)].

    Multi-ref (OpenSubject-style scene+subject conditioning): each reference gets its own
    RoPE frame index (1..N) with its own h,w grid — ref_1 (the scene) stays h/w-aligned with
    the target for preservation; extra refs (subject photos) are spatially independent
    identity/appearance references, distinguished purely by their frame index.
    """
    patch = model.config.patch
    b, c, h, w = target_latents.shape

    n = model.config.txtlayers
    context = context.reshape(context.shape[0], context.shape[1], n, context.shape[-1] // n)

    if not isinstance(source_latents, (list, tuple)):
        source_latents = [source_latents]
    tgt_tok, tgt_pos, tgt_mask = _img_tokens_and_pos(target_latents, patch, frame=0)
    src_toks, src_poss, src_masks, src_len = [], [], [], 0
    for i, sl in enumerate(source_latents):
        tok, pos_i, mask_i = _img_tokens_and_pos(sl, patch, frame=i + 1)
        src_toks.append(tok); src_poss.append(pos_i); src_masks.append(mask_i)
        src_len += tok.shape[1]

    img_tokens = torch.cat(src_toks + [tgt_tok], dim=1)
    txtlen = context.shape[1]
    txtpos = torch.zeros(b, txtlen, 3, device=target_latents.device)
    pos = torch.cat([txtpos] + src_poss + [tgt_pos], dim=1)
    mask = torch.cat([text_mask.to(target_latents.device).bool()] + src_masks + [tgt_mask], dim=1)

    out = model(
        img=img_tokens,
        context=context,
        t=t,
        pos=pos,
        mask=mask,
        # Ref latent values are clean already. This also gives their transformer
        # modulation the clean endpoint t=0, while text/target stay at sampled t.
        ref_token_count=src_len,
    )  # [refs... ; tgt_out]
    target_out = out[:, src_len:]  # keep only the target tokens
    velocity = rearrange(
        target_out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch, h=h // patch, w=w // patch,
    )
    return velocity


def predict_velocity_edit_varlen(
    model: SingleStreamDiT,
    target_latents: torch.Tensor,
    source_latents: Sequence[Sequence[torch.Tensor]],
    t: torch.Tensor,
    context: Sequence[torch.Tensor],
) -> torch.Tensor:
    """FA4-packed edit prediction for B>1 samples with ragged references.

    The target remains a normal bucketed ``(B,C,H,W)`` tensor because ai-toolkit's
    noise scheduler and flow-matching loss are dense.  References are deliberately
    *not* batched: each sample may have a different number of sources and every
    source may have its own latent H/W.  Their text contexts are kept unpadded too.
    """
    batch_size, channels, height, width = target_latents.shape
    if len(source_latents) != batch_size or len(context) != batch_size:
        raise ValueError(
            "ragged edit batch needs one reference list and text context per target sample "
            f"(got refs={len(source_latents)}, context={len(context)}, batch={batch_size})"
        )

    patch = model.config.patch
    layer_count = model.config.txtlayers
    image_tokens, image_positions, ref_token_counts, text_contexts = [], [], [], []
    target_token_lengths = []
    for sample, (refs, text_features) in enumerate(zip(source_latents, context)):
        if not refs:
            raise ValueError(
                f"ragged edit sample {sample} has no reference latent; mix edit and T2I "
                "examples in separate datasets/batches."
            )
        ref_tokens, ref_positions = [], []
        for frame, ref in enumerate(refs, start=1):
            if ref.ndim == 3:
                ref = ref.unsqueeze(0)
            if ref.ndim != 4 or ref.shape[0] != 1:
                raise ValueError(
                    f"ragged reference {frame} for sample {sample} must be (1,C,H,W), "
                    f"got {tuple(ref.shape)}"
                )
            tokens, positions, _ = _img_tokens_and_pos(ref, patch, frame=frame)
            ref_tokens.append(tokens.squeeze(0))
            ref_positions.append(positions.squeeze(0))

        target_tokens, target_positions, _ = _img_tokens_and_pos(
            target_latents[sample:sample + 1], patch, frame=0
        )
        target_tokens = target_tokens.squeeze(0)
        target_positions = target_positions.squeeze(0)
        ref_count = sum(tokens.shape[0] for tokens in ref_tokens)
        image_tokens.append(torch.cat(ref_tokens + [target_tokens], dim=0))
        image_positions.append(torch.cat(ref_positions + [target_positions], dim=0))
        ref_token_counts.append(ref_count)
        target_token_lengths.append(target_tokens.shape[0])

        if text_features.ndim != 2 or text_features.shape[-1] % layer_count:
            raise ValueError(
                f"ragged text context for sample {sample} must be (L, n_layers*hidden), "
                f"got {tuple(text_features.shape)}"
            )
        text_contexts.append(text_features.reshape(
            text_features.shape[0], layer_count, text_features.shape[-1] // layer_count
        ))

    predictions = model.forward_varlen(
        image_tokens=image_tokens,
        context=text_contexts,
        t=t,
        image_positions=image_positions,
        ref_token_counts=ref_token_counts,
    )
    expected_tokens = (height // patch) * (width // patch)
    if any(length != expected_tokens for length in target_token_lengths):
        # This should be impossible while ai-toolkit buckets targets, but spell out
        # the boundary rather than silently reinterpreting a ragged target batch.
        raise ValueError(
            "ragged target grids are not supported by ai-toolkit's dense loss. "
            "Enable dataset buckets so every target in this batch has the same H/W."
        )
    velocities = [
        rearrange(
            pred, "(h w) (c ph pw) -> c (h ph) (w pw)",
            ph=patch, pw=patch, h=height // patch, w=width // patch,
        )
        for pred in predictions
    ]
    return torch.stack(velocities, dim=0).to(target_latents.dtype)


class Krea2EditModel(Krea2Model):
    arch = "krea2_edit"

    def __init__(self, device, model_config: ModelConfig, dtype="bf16",
                 custom_pipeline=None, noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        # route the control image to get_prompt_embeds (semantic grounding)
        self.encode_control_in_text_embeddings = True
        self._control_latents = None  # Tensor (single-ref) or List[Tensor] (multi-ref)
        # Every image keeps its own independent 2D grid. The raw reference is never
        # cropped or target-fitted; it is resized by at most half a patch per axis to
        # the nearest VAE/DiT 16px lattice before latent patchification.
        model_kwargs = model_config.model_kwargs or {}
        self.use_raw_control_images = True
        if "fit_refs" in model_kwargs:
            print("[krea2_edit] NOTE: model_kwargs.fit_refs is obsolete and ignored; "
                  "references now use independent patch-aligned grids (no crop).")
        print("[krea2_edit] PATCH-ALIGNED refs ENABLED (independent RoPE grids; nearest-16px resize)")
        # Grounding policy, printed once so every run is self-documenting.
        _g_max, _g_jit = grounding_settings()
        print(f"[krea2_edit] grounding: GROUNDING_MAX_PX={_g_max} GROUNDING_JITTER_MIN={_g_jit} "
              f"(recipe defaults 768/384; 0 = off/native resolution)")
        if _g_max <= 0:
            print("[krea2_edit] WARNING: grounding cap disabled -> the VLM sees native-resolution "
                  "references, which is NOT the released recipe (node default grounding_px=768).")
        # ai-toolkit consults this flag in its cached-text path. Always advertise
        # multi-reference support so every configured control image reaches Qwen3-VL,
        # not just the first one. The appearance path already handles an arbitrary N.
        self.has_multiple_control_images = True
        if "multi_ref" in model_kwargs:
            print("[krea2_edit] NOTE: model_kwargs.multi_ref is obsolete; arbitrary "
                  "reference counts are enabled by default.")
        print("[krea2_edit] multi-reference conditioning ENABLED (frames 1..N)")
        self._validate_raw_control_training_config()

    # ------------------------------------------------------------------
    # Guards: refuse configurations the inference nodes cannot reproduce
    # ------------------------------------------------------------------
    @staticmethod
    def _find_owning_train_process():
        """Best-effort handle on the ai-toolkit train process constructing this model.

        ai-toolkit hands a model class only its ModelConfig — there is no official
        view of train/dataset config from here — but the process instance is a local
        of the calling frame (``BaseSDTrainProcess.load_model`` -> ``ModelClass(...)``),
        so the guards below can read train_config and the dataset configs. Returns
        None when we are not being constructed by a train process (plain import,
        third-party harness); the guards then no-op and the README restrictions
        apply on the honor system.
        """
        import inspect
        frame = inspect.currentframe()
        try:
            while frame is not None:
                obj = frame.f_locals.get("self", None)
                if (obj is not None and hasattr(obj, "train_config")
                        and hasattr(obj, "dataset_configs")):
                    return obj
                frame = frame.f_back
        except Exception:
            return None
        finally:
            del frame
        return None

    @staticmethod
    def _control_paths_of(ds) -> List[str]:
        """Normalized list of control_path entries declared by a DatasetConfig."""
        cp = getattr(ds, "control_path", None)
        if cp is None or cp == "":
            return []
        return list(cp) if isinstance(cp, (list, tuple)) else [cp]

    @classmethod
    def _audit_pairs(cls, ds):
        """Filename-only audit of one dataset's target/source stem pairing.

        Mirrors ai-toolkit's own resolution exactly (``AiToolkitDataset.__init__``
        builds the target list with ``os.walk`` + ``image_extensions``, skipping
        dotfiles and ``_controls/``; ``ControlFileItemDTOMixin.__init__`` then looks
        for ``<control_dir>/<target stem><ext>`` for ext in ``img_ext_list``), so the
        counts here are what the dataloader will actually resolve.

        The dataset objects themselves do not exist yet at model-construction time —
        ``BaseSDTrainProcess.run()`` builds the model (``self.sd = ModelClass(...)``)
        long before ``get_dataloader_from_datasets`` — so we resolve from the dataset
        *config* instead. Only directory entries are read; no image or caption file is
        ever opened.

        Returns ``None`` when the audit does not apply (json-manifest dataset,
        ``control_from_same_folder``, missing folder), else a dict with the counts.
        """
        ctrl_dirs = cls._control_paths_of(ds)
        if not ctrl_dirs or getattr(ds, "control_from_same_folder", False):
            return None
        root = getattr(ds, "dataset_path", None) or getattr(ds, "folder_path", None)
        if not root or not os.path.isdir(root):
            return None  # json manifest or a bad path — upstream reports that itself
        exts = (".jpg", ".jpeg", ".png", ".webp")
        targets = []
        for dirpath, _dirs, files in os.walk(root):
            if os.path.basename(dirpath) == "_controls":
                continue
            for fn in files:
                if fn.lower().endswith(exts) and not fn.startswith("."):
                    targets.append(fn)
        missing_all, partial, per_dir = [], [], {d: 0 for d in ctrl_dirs}
        for fn in targets:
            stem = os.path.splitext(fn)[0]
            hits = 0
            for d in ctrl_dirs:
                if any(os.path.exists(os.path.join(d, stem + e)) for e in exts):
                    hits += 1
                else:
                    per_dir[d] += 1
            if hits == 0:
                missing_all.append(fn)
            elif hits < len(ctrl_dirs):
                partial.append(fn)
        return {
            "root": root, "ctrl_dirs": ctrl_dirs, "total": len(targets),
            "missing_all": missing_all, "partial": partial, "per_dir": per_dir,
        }

    def _validate_raw_control_training_config(self):
        """Hard-fail configs that silently train something the nodes cannot reproduce.

        Runs once, from ``__init__`` — i.e. before the base weights, the text encoder
        and the latent/text caches are loaded, so a broken config costs seconds rather
        than a full model load (or, worse, a whole run that only looks healthy).
        """
        proc = self._find_owning_train_process()
        if proc is None:
            print("[krea2_edit] NOTE: train/dataset config not visible from the model "
                  "class; the startup guards (batch_size, flip augmentation, "
                  "source-image pairing, reference count, text-embedding caching, "
                  "unload_text_encoder) were skipped. See README, 'Not supported'.")
            return
        datasets = list(getattr(proc, "dataset_configs", None) or [])

        # --- guards for native-size raw control images ------------------------------
        if self.use_raw_control_images:
            batch_size = int(getattr(proc.train_config, "batch_size", 1) or 1)
            if batch_size > 1:
                unbucketed = [
                    getattr(d, "folder_path", "<dataset>")
                    for d in datasets
                    if self._control_paths_of(d) and not getattr(d, "buckets", False)
                ]
                if unbucketed:
                    raise ValueError(
                        "krea2_edit: ragged references support batch_size > 1, but target "
                        "latents still use ai-toolkit's dense flow-matching loss.\n"
                        "  Enable buckets: true for every edit dataset so each batch has one "
                        "target H×W; source/reference H×W may differ freely.\n"
                        f"  Unbucketed edit dataset(s): {unbucketed}"
                    )
                print(
                    f"[krea2_edit] batch_size={batch_size}: ragged reference mode ENABLED "
                    "(FlashAttention-4 varlen; target grids must be bucketed)",
                    flush=True,
                )
            flipped = [
                getattr(d, "folder_path", "<dataset>")
                for d in datasets
                if getattr(d, "flip_x", False) or getattr(d, "flip_y", False)
            ]
            if flipped:
                raise ValueError(
                    "krea2_edit: flip augmentation (flip_x / flip_y) is not supported with the "
                    "patch-aligned independent-reference geometry.\n"
                    "  ai-toolkit flips the TARGET image but not the raw control (reference) "
                    "images, so every flipped pair is silently desynced — the LoRA is trained "
                    "on mirrored supervision.\n"
                    f"  Offending dataset(s): {flipped}\n"
                    "  Set flip_x: false / flip_y: false, or pre-flip your pairs offline."
                )

        # --- guards that apply to all edit batches ----------------------------------
        # The whole arch grounds the instruction on the reference images inside the
        # text encoder. Unloading the TE makes ai-toolkit fall back to a cached BLANK
        # embedding for every step (SDTrainer: `if unload_text_encoder or
        # is_caching_text_embeddings` -> `cached_blank_embeds` when the batch has no
        # cached embeds), i.e. the model trains with no instruction and no grounding.
        if getattr(proc.train_config, "unload_text_encoder", False):
            raise ValueError(
                "krea2_edit: train.unload_text_encoder: true is not supported.\n"
                "  This architecture encodes the reference image(s) INSIDE the text "
                "encoder (the instruction is image-grounded), so the text encoder must "
                "stay resident. With it unloaded and no cached embeddings, ai-toolkit "
                "feeds a blank embedding on every step — the run looks healthy and "
                "trains on no conditioning at all.\n"
                "  Set train.unload_text_encoder: false. To save the TE's VRAM instead, "
                "use dataset cache_text_embeddings: true (which freezes one grounding "
                "scale — see the README VRAM section for the tradeoff)."
            )

        for ds in datasets:
            ctrl_dirs = self._control_paths_of(ds)
            folder = getattr(ds, "folder_path", None) or getattr(ds, "dataset_path", "<dataset>")
            if not ctrl_dirs:
                continue  # targets-only (plain T2I regularization) — legitimate

            # A target with no stem-matched source trains as plain T2I, silently.
            report = self._audit_pairs(ds)
            if report is None:
                continue
            if report["missing_all"]:
                n, m = len(report["missing_all"]), report["total"]
                ex = ", ".join(sorted(report["missing_all"])[:5])
                more = "" if n <= 5 else f" (+{n - 5} more)"
                raise ValueError(
                    f"krea2_edit: {n} of {m} target images in dataset '{report['root']}' have "
                    "no stem-matched source image.\n"
                    "  ai-toolkit resolves a control image by filename stem: for target "
                    "<stem>.<ext> it looks for <control_dir>/<stem>.{jpg,jpeg,png,webp}. When "
                    "nothing matches, the item loses its control image and trains as PLAIN "
                    "TEXT-TO-IMAGE — no reference tokens, no image-grounded text encode. The "
                    "run looks healthy and the LoRA quietly learns the wrong thing.\n"
                    f"  control_path searched: {report['ctrl_dirs']}\n"
                    f"  Example unpaired target filenames: {ex}{more}\n"
                    "  Fix: add the missing source images (stems must match exactly, "
                    "extensions need not), or move the unpaired targets into a separate "
                    "targets-only dataset entry with no control_path if you really do want "
                    "them as T2I regularization."
                )
            if report["partial"]:
                n, m = len(report["partial"]), report["total"]
                ex = ", ".join(sorted(report["partial"])[:5])
                more = "" if n <= 5 else f" (+{n - 5} more)"
                per_dir = ", ".join(f"{d}: {c} missing" for d, c in report["per_dir"].items())
                bar = "!" * 78
                print(bar)
                print(f"[krea2_edit] WARNING: {n} of {m} targets in '{report['root']}' match "
                      "SOME but not all control_path entries.")
                print(f"[krea2_edit]   per control_path -> {per_dir}")
                print(f"[krea2_edit]   examples: {ex}{more}")
                print("[krea2_edit]   Those items train with fewer references, and reference "
                      "ORDER shifts: a target")
                print("[krea2_edit]   matched only by the SECOND control_path gets it as "
                      "reference #1 (frame 1).")
                print("[krea2_edit]   Complete the pairs, or split them into their own "
                      "single-reference dataset entry.")
                print(bar)

    def load_model(self):
        super().load_model()
        ckpt_n = self.model_config.model_kwargs.get("checkpoint_every_n", 1)
        if ckpt_n > 1:
            self.transformer.checkpoint_every_n = ckpt_n
            self.print_and_status_update(
                f"Selective checkpointing: every {ckpt_n}th block "
                f"({len(self.transformer.blocks) // ckpt_n} of {len(self.transformer.blocks)} checkpointed)"
            )

    # ------------------------------------------------------------------
    # In-training previews: unsupported for the edit arch
    # ------------------------------------------------------------------
    _NO_SAMPLING_MSG = (
        "krea2_edit: in-training sample previews are not supported for this architecture.\n"
        "  ai-toolkit's sampling path renders the inherited plain-T2I pipeline: no "
        "reference token blocks (frame>=1) and no image-grounded Qwen3-VL encode, so the "
        "previews show what the LoRA does WITHOUT the edit conditioning — misleading at "
        "best.\n"
        "  Set train.disable_sampling: true and evaluate checkpoints in ComfyUI with the "
        "comfyui-krea2edit nodes."
    )

    def get_generation_pipeline(self):
        raise NotImplementedError(self._NO_SAMPLING_MSG)

    def generate_single_image(self, *args, **kwargs):
        raise NotImplementedError(self._NO_SAMPLING_MSG)

    def _load_text_encoder(self):
        """Same as Krea2Model but KEEP the Qwen3-VL vision tower.

        The T2I base drops ``text_encoder.model.visual`` to save VRAM (it only ever
        encodes text). The editor grounds the instruction on the source image, so the
        vision tower must survive.
        """
        from . import krea2 as _k
        dtype = self.torch_dtype
        te_path = self.model_config.model_kwargs.get("text_encoder_path", _k.QWEN3_VL_PATH)
        self.print_and_status_update(
            f"Loading Qwen3-VL text encoder (with vision tower) from {te_path}"
        )
        tokenizer = _k.AutoTokenizer.from_pretrained(
            te_path, max_length=self.max_text_length, token=_k.HF_TOKEN
        )
        processor = _k.Qwen2TokenizerFast.from_pretrained(
            te_path, max_length=self.max_text_length, token=_k.HF_TOKEN
        )
        text_encoder = _k.Qwen3VLForConditionalGeneration.from_pretrained(
            te_path, torch_dtype=dtype, token=_k.HF_TOKEN
        )
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        _k.flush()
        return tokenizer, processor, text_encoder

    # --- appearance path: stash the source latent for the prediction step ---
    @staticmethod
    def _resize_reference_to_grid(c):
        """Resize a reference to the nearest VAE / DiT patch grid.

        Krea's VAE is /8 and the DiT patch is 2x2 latent cells, so each source image
        must be divisible by 16 px. Resize H and W independently to the closest grid
        (rather than padding or cropping) and retain the source's own RoPE origin.
        """
        h, w = c.shape[-2:]
        aligned_h = max(16, ((h + 8) // 16) * 16)
        aligned_w = max(16, ((w + 8) // 16) * 16)
        if (aligned_h, aligned_w) != (h, w):
            c = F.interpolate(c, size=(aligned_h, aligned_w), mode="bilinear",
                              align_corners=False)
        return c

    def condition_noisy_latents(self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"):
        with torch.no_grad():
            ctrl = getattr(batch, "control_tensor", None)
            ctrl_list = getattr(batch, "control_tensor_list", None)
            if ctrl is None and ctrl_list:
                # Ragged raw-control path: the extension collator turns every
                # sample into ``[[ref_1, ...], ...]`` before ai-toolkit's stock
                # DTO can concatenate them.  Each source gets a separate VAE call
                # because neither its H/W nor its reference count is shared.
                batch_size = latents.shape[0]
                if (len(ctrl_list) == batch_size
                        and all(isinstance(refs, (list, tuple)) for refs in ctrl_list)):
                    refs_per_item = ctrl_list
                elif batch_size == 1:
                    # Backwards-compatible direct call / old ai-toolkit list form.
                    refs_per_item = [ctrl_list]
                elif len(ctrl_list) == batch_size and all(torch.is_tensor(ref) for ref in ctrl_list):
                    refs_per_item = [[ref] for ref in ctrl_list]
                else:
                    raise ValueError(
                        "krea2_edit: malformed ragged control batch; expected one reference "
                        f"list per target (refs={len(ctrl_list)}, batch={batch_size})."
                    )

                all_lats = []
                for refs in refs_per_item:
                    item_lats = []
                    for c in refs:
                        if c.dim() == 3:
                            c = c.unsqueeze(0)
                        c = (c * 2 - 1).to(self.vae_device_torch, dtype=self.torch_dtype)
                        c = self._resize_reference_to_grid(c)
                        item_lats.append(self.encode_images(c).to(latents.device, latents.dtype))
                    all_lats.append(item_lats)
                self._control_latents = all_lats
                if not getattr(self, "_edit_path_logged", False):
                    print(f"[krea2_edit] ragged patch-aligned refs: "
                          f"{[[tuple(l.shape[-2:]) for l in item] for item in all_lats]} "
                          f"for target {tuple(latents.shape[-2:])}", flush=True)
                    self._edit_path_logged = True
                return latents.detach()
            if ctrl is not None:
                if ctrl.dim() == 5:  # multi-ref: (B, N, C, H, W) -> list of per-ref latents
                    lats = []
                    for i in range(ctrl.shape[1]):
                        c_i = (ctrl[:, i] * 2 - 1).to(self.vae_device_torch, dtype=self.torch_dtype)
                        c_i = self._resize_reference_to_grid(c_i)
                        lats.append(self.encode_images(c_i).to(latents.device, latents.dtype))
                    self._control_latents = lats
                else:
                    c = (ctrl * 2 - 1).to(self.vae_device_torch, dtype=self.torch_dtype)
                    c = self._resize_reference_to_grid(c)
                    self._control_latents = self.encode_images(c).to(latents.device, latents.dtype)
            else:
                self._control_latents = None
        if not getattr(self, "_edit_path_logged", False):
            nref = (len(self._control_latents) if isinstance(self._control_latents, list)
                    else (1 if self._control_latents is not None else 0))
            print(f"[krea2_edit] control_tensor present: {ctrl is not None} (refs={nref}) "
                  f"-> edit path {'ACTIVE' if self._control_latents is not None else 'INACTIVE (plain T2I!)'}")
            self._edit_path_logged = True
        return latents.detach()

    def get_noise_prediction(self, latent_model_input, timestep, text_embeddings, **kwargs):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        t = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != latent_model_input.shape[0]:
            t = t.expand(latent_model_input.shape[0])
        li = latent_model_input.to(self.device_torch, self.torch_dtype)
        ragged_controls = (
            isinstance(self._control_latents, list)
            and len(self._control_latents) == li.shape[0]
            and all(isinstance(refs, (list, tuple)) for refs in self._control_latents)
        )
        # torch.compile the (LoRA-attached) transformer lazily on the first forward: by now the
        # LoRA is applied, so the compiled graph captures it. Cached -> reference-stable, compiled
        # once. Opt-in via KREA2_COMPILE=1 (needs TRITON_PTXAS_PATH=CUDA-13 ptxas for sm_121a);
        # any compile error falls back to eager so the run never dies on it.
        m = self.transformer
        if (os.environ.get("KREA2_COMPILE") == "1" and not getattr(self, "_compiled", False)
                and not (ragged_controls and li.shape[0] > 1)):
            # Block-level compile (in-place): compile each transformer block, not the whole 12.9B
            # graph -- whole-model compile spiked ~+58GB and the watchdog killed it at batch>1. Per-
            # block graphs compile in a fraction of the memory and still fuse the RoPE/norm/SwiGLU.
            # Done here (post first-forward) so the LoRA is attached and captured. Eager fallback.
            try:
                _mode = os.environ.get("KREA2_COMPILE_MODE", "default")
                _blocks = self.transformer.blocks
                for _i in range(len(_blocks)):
                    _blocks[_i] = torch.compile(_blocks[_i], mode=_mode)
                print(f"[krea2_edit] block-compiled {len(_blocks)} transformer blocks", flush=True)
            except Exception as _e:
                print(f"[krea2_edit] block-compile FAILED -> eager: {_e}", flush=True)
            self._compiled = True
        # self.transformer.blocks are now compiled in-place (m is self.transformer)
        if self._control_latents is None:
            context, text_mask = pad_text_features(
                text_embeddings.text_embeds, self.device_torch, self.torch_dtype
            )
            return predict_velocity(m, li, t, context, text_mask)

        if ragged_controls and li.shape[0] > 1:
            raw_context = text_embeddings.text_embeds
            if torch.is_tensor(raw_context):
                if raw_context.dim() == 2 and li.shape[0] == 1:
                    raw_context = [raw_context]
                elif raw_context.dim() == 3:
                    raw_context = [raw_context[i] for i in range(raw_context.shape[0])]
                else:
                    raise ValueError(
                        "krea2_edit: packed edit context must be a list of (L,F) tensors "
                        "or a (B,L,F) tensor."
                    )
            else:
                raw_context = list(raw_context)
            contexts = [c.to(self.device_torch, dtype=self.torch_dtype) for c in raw_context]
            return predict_velocity_edit_varlen(
                m, li, self._control_latents, t, contexts
            )

        context, text_mask = pad_text_features(
            text_embeddings.text_embeds, self.device_torch, self.torch_dtype
        )

        # A single-item ragged DTO can retain the established SDPA path. It avoids
        # imposing FlashAttention-4 as a dependency on ordinary batch_size: 1 runs.
        if ragged_controls:
            src = [s.to(self.device_torch, self.torch_dtype) for s in self._control_latents[0]]
            return predict_velocity_edit(m, li, src, t, context, text_mask)
        def _fit(s):
            s = s.to(self.device_torch, self.torch_dtype)
            if s.shape[0] != li.shape[0]:
                s = s.expand(li.shape[0], *s.shape[1:])
            return s
        if isinstance(self._control_latents, list):
            src = [_fit(s) for s in self._control_latents]
        else:
            src = _fit(self._control_latents)
        return predict_velocity_edit(m, li, src, t, context, text_mask)

    # --- semantic path: image-grounded Qwen3-VL encode ---
    def get_prompt_embeds(self, prompt, control_images=None) -> "AdvancedPromptEmbeds":
        if control_images is None:
            return super().get_prompt_embeds(prompt)
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        # TE_BATCHED=1: ONE right-padded Qwen3-VL forward for the whole batch instead of the
        # serial per-item loop below (~0.65s/item on H100 -> dominates on-the-fly step time).
        # Numerics validated vs single-item (relL2<0.05, Spark 2026-07-01 + local re-validation).
        # Single-ref 4D only; multi-ref always takes the serial path.
        if (os.environ.get("TE_BATCHED", "0") == "1" and torch.is_tensor(control_images)
                and control_images.dim() == 4 and control_images.shape[0] == len(prompt)):
            imgs = [control_images[i] for i in range(control_images.shape[0])]
            feats = self._encode_image_prompt_batch(
                [p if isinstance(p, str) else "" for p in prompt], imgs)
            feats = [f.reshape(f.shape[0], -1).to(self.torch_dtype) for f in feats]
            return AdvancedPromptEmbeds(text_embeds=feats)

        # Normalize every input shape into per-item image LISTS (multi-ref grounding = one
        # vision block per reference image in the same user turn):
        #   4D (B,C,H,W)   -> [[img]] per item          (single-ref training batch)
        #   5D (B,N,C,H,W) -> [[img_1..img_N]] per item (multi-ref training batch)
        #   list           -> one item, list of refs    (caching mixin, has_multiple_control_images)
        if torch.is_tensor(control_images):
            if control_images.dim() == 5:
                per_item = [[control_images[i, j] for j in range(control_images.shape[1])]
                            for i in range(control_images.shape[0])]
            elif control_images.dim() == 4 and control_images.shape[0] == len(prompt):
                per_item = [[control_images[i]] for i in range(control_images.shape[0])]
            else:
                per_item = [[control_images.squeeze(0) if control_images.dim() == 4 else control_images]] * len(prompt)
        elif isinstance(control_images, (list, tuple)):
            if control_images and isinstance(control_images[0], (list, tuple)):
                # raw-control path: list of per-item ref LISTS (ragged sizes)
                per_item = [[(im.squeeze(0) if torch.is_tensor(im) and im.dim() == 4 else im)
                             for im in item] for item in control_images]
                if len(per_item) == 1 and len(prompt) > 1:
                    per_item = per_item * len(prompt)
            else:
                refs = [(im.squeeze(0) if torch.is_tensor(im) and im.dim() == 4 else im) for im in control_images]
                per_item = [refs] * len(prompt)
        else:
            per_item = [[control_images]] * len(prompt)

        feats = []
        for i, p in enumerate(prompt):
            # ai-toolkit passes neg=False for an unset negative prompt -> treat any
            # non-string (False/None) as an empty instruction (null text + source image).
            p = p if isinstance(p, str) else ""
            f = self._encode_image_prompt(p, per_item[i])
            feats.append(f.reshape(f.shape[0], -1).to(self.torch_dtype))
        return AdvancedPromptEmbeds(text_embeds=feats)

    def _grounding_pil(self, image: torch.Tensor):
        """Control tensor (C,H,W) 0..1 -> PIL for the VLM, with env-gated downscale.

        The grounding branch is SEMANTIC — fine detail reaches the DiT through the VAE
        source-latent tokens, not through Qwen3-VL — so capping the grounding image cuts
        vision tokens ~(native/cap)^2 with little semantic loss, shrinking the DiT's text
        stream (faster steps) and the TE forward (faster on-the-fly encoding).

        Defaults are the released recipe (max 768 / jitter min 384) and match the
        inference node's grounding_px default; the env vars only override them.

        GROUNDING_MAX_PX     (int, default 768, 0=off): cap the longest side fed to Qwen3-VL.
        GROUNDING_JITTER_MIN (int, default 384, 0=off): with MAX_PX set, sample the cap per
            call from uniform[MIN, MAX_PX] -> the model learns granularity/length
            robustness, so inference may freely use other grounding scales without a
            train/infer mismatch.
        """
        from PIL import Image
        import numpy as np
        import random
        arr = (image.detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr.transpose(1, 2, 0))
        cap, jmin = grounding_settings()
        if cap > 0:
            tgt = random.randint(jmin, cap) if 0 < jmin < cap else cap
            if max(pil.size) > tgt:
                s = tgt / max(pil.size)
                pil = pil.resize(
                    (max(1, round(pil.size[0] * s)), max(1, round(pil.size[1] * s))),
                    Image.LANCZOS)
        return pil

    def _text_only_vlm_context(
        self,
        hiddens: torch.Tensor,
        input_ids: torch.Tensor,
        end: Optional[int] = None,
    ) -> torch.Tensor:
        """Remove Qwen3-VL image-patch states before sending context to the DiT.

        Qwen3-VL replaces every ``<|image_pad|>`` position in its input sequence
        with a visual embedding. The VLM still processes those positions, so the
        retained instruction and assistant states remain image-grounded through
        causal attention. Only the visual-patch states are removed from the DiT
        context. ``<|vision_start|>`` and ``<|vision_end|>`` are language-model
        delimiter tokens and are intentionally retained.

        ``hiddens`` is one sample with shape ``(S, n_layers, hidden)`` and
        ``input_ids`` is its matching unpadded token sequence.
        """
        if hiddens.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "krea2_edit: Qwen3-VL hidden-state and input-id sequence lengths differ; "
                "cannot safely remove visual tokens."
            )

        image_token_id = getattr(self.text_encoder.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self._mm_processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_token_id is None or image_token_id < 0:
            raise ValueError(
                "krea2_edit: could not resolve Qwen3-VL's <|image_pad|> token id; "
                "refusing to pass unfiltered visual states to the DiT."
            )

        stop = hiddens.shape[0] if end is None else end
        states = hiddens[PROMPT_TEMPLATE_ENCODE_START_IDX:stop]
        ids = input_ids[PROMPT_TEMPLATE_ENCODE_START_IDX:stop]
        return states[ids != image_token_id]

    @torch.no_grad()
    def _encode_image_prompt_batch(self, prompts: List[str], images: List[torch.Tensor]) -> List[torch.Tensor]:
        """Batched twin of _encode_image_prompt: one padded forward, per-item valid slice.

        Right padding keeps the fixed START_IDX system prefix aligned at the front of every
        row (causal attention: real tokens never attend to the pads that follow them), and
        the attention_mask valid-length slice drops the pads from the tapped hidden states.
        Returns a list of text-only DiT contexts with shape ``(L_i, n_layers, d)``.
        Qwen3-VL image-patch states are removed after multimodal encoding, while
        retained language-token states stay image-grounded.
        """
        dev = self.text_encoder.device
        if getattr(self, "_mm_processor", None) is None:
            from transformers import AutoProcessor
            te_path = self.model_config.model_kwargs.get(
                "text_encoder_path", "Qwen/Qwen3-VL-4B-Instruct")
            self._mm_processor = AutoProcessor.from_pretrained(te_path)
        pils = [self._grounding_pil(im) for im in images]
        vis = "<|vision_start|><|image_pad|><|vision_end|>"
        texts = [PROMPT_TEMPLATE_ENCODE_PREFIX + vis + p + PROMPT_TEMPLATE_ENCODE_SUFFIX
                 for p in prompts]
        tok = self._mm_processor.tokenizer
        old_side = tok.padding_side
        tok.padding_side = "right"
        try:
            inputs = self._mm_processor(
                text=texts, images=pils, padding=True, return_tensors="pt",
            ).to(dev, non_blocking=True)
        finally:
            tok.padding_side = old_side
        states = self.text_encoder(**inputs, output_hidden_states=True)
        hiddens = torch.stack([states.hidden_states[i] for i in SELECT_LAYERS], dim=2)  # (B,S,n,d)
        mask = inputs["attention_mask"]
        feats = []
        for b in range(len(prompts)):
            valid = int(mask[b].sum().item())
            feats.append(self._text_only_vlm_context(
                hiddens[b, :valid], inputs["input_ids"][b, :valid]
            ))
        return feats

    @torch.no_grad()
    def _encode_image_prompt(self, prompt: str, image) -> torch.Tensor:
        """Feed (reference image(s) + instruction) through Qwen3-VL; tap SELECT_LAYERS.

        `image` may be a single (C,H,W) tensor or a LIST of them (multi-ref): each
        reference contributes its own <|vision_start|><|image_pad|><|vision_end|> block
        in the same user turn, in order (scene first, subject refs after) — mirroring
        the frame-index order of the token blocks in predict_velocity_edit. The VLM
        sees all image tokens, but their hidden states are removed before the result
        is handed to the DiT; only image-grounded language-token states remain.
        """
        dev = self.text_encoder.device
        # krea2's self.processor is a TEXT tokenizer; load a real multimodal
        # processor (with image_processor) lazily for the vision path.
        if getattr(self, "_mm_processor", None) is None:
            from transformers import AutoProcessor
            te_path = self.model_config.model_kwargs.get(
                "text_encoder_path", "Qwen/Qwen3-VL-4B-Instruct"
            )
            self._mm_processor = AutoProcessor.from_pretrained(te_path)
        images = image if isinstance(image, (list, tuple)) else [image]
        # tensors (C,H,W) 0..1 -> PILs (env-gated grounding downscale/jitter inside)
        pils = [self._grounding_pil(im) for im in images]
        vis = "<|vision_start|><|image_pad|><|vision_end|>" * len(pils)
        prefix = PROMPT_TEMPLATE_ENCODE_PREFIX + vis
        inputs = self._mm_processor(
            text=[prefix + prompt + PROMPT_TEMPLATE_ENCODE_SUFFIX],
            images=pils, return_tensors="pt",
        ).to(dev, non_blocking=True)
        states = self.text_encoder(**inputs, output_hidden_states=True)
        hiddens = torch.stack([states.hidden_states[i] for i in SELECT_LAYERS], dim=2)
        return self._text_only_vlm_context(hiddens[0], inputs["input_ids"][0])
