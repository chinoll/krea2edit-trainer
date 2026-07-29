from .krea2 import Krea2Model
from .krea2_edit import Krea2EditModel

# ai-toolkit discovers models from packages in `extensions/` via this attribute.
AI_TOOLKIT_MODELS = [Krea2Model, Krea2EditModel]

__all__ = ["Krea2Model", "Krea2EditModel", "AI_TOOLKIT_MODELS"]
