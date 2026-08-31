from .krea2_edit import Krea2EditModel, install_ragged_control_collator

# This runs while ai-toolkit discovers extensions, before it builds the training
# DataLoader.  It replaces only the collator used by this edit architecture; plain
# ai-toolkit training keeps its normal dense-control path.
install_ragged_control_collator()

# Register ONLY the edit arch: upstream ai-toolkit already ships Krea 2 T2I
# ("krea2"), and registering a second copy would collide with it. Our vendored
# krea2.py stays as the (unregistered) base class the edit model builds on.
AI_TOOLKIT_MODELS = [Krea2EditModel]

__all__ = ["Krea2EditModel", "AI_TOOLKIT_MODELS"]
