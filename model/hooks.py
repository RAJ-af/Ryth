"""Forward hooks for research / debugging.

Har TransformerBlock ke andar 4 hook points hote hain:
    before_attention · after_attention · before_ffn · after_ffn

In par callbacks register karo — activations inspect karo, ya modify bhi kar do
(agar callback tensor return kare to wo aage use hota hai). Default: no-op, zero
overhead. State-dict me kuch nahi jaata (plain objects, nn.Module nahi).

    def probe(t, **ctx):
        print(ctx["layer"], t.shape, t.std().item())
    model.register_hook("after_attention", probe)
"""

from __future__ import annotations

HOOK_POINTS = ("before_attention", "after_attention", "before_ffn", "after_ffn")


class HookPoint:
    """Ek named hook site. Callbacks (tensor, **ctx) -> optional tensor."""

    def __init__(self, name: str):
        self.name = name
        self._fns = []

    def register(self, fn):
        self._fns.append(fn)
        return fn

    def clear(self):
        self._fns.clear()

    def __call__(self, tensor, **ctx):
        for fn in self._fns:
            out = fn(tensor, **ctx)
            if out is not None:                # callback modify kar sakta hai
                tensor = out
        return tensor


class BlockHooks:
    """Ek block ke chaar hook points ka container (plain object, not a Module)."""

    def __init__(self):
        for name in HOOK_POINTS:
            setattr(self, name, HookPoint(name))

    def clear(self):
        for name in HOOK_POINTS:
            getattr(self, name).clear()
