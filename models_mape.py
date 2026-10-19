from __future__ import annotations

from models5 import *
import models5 as _models5


class GIN(_models5.GIN):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
