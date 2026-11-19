"""Langfuse tracing. An agent run without a tracer is refused unless explicitly allowed."""

from __future__ import annotations

import os
from contextlib import contextmanager


class UntracedRunError(RuntimeError):
    pass


class _NullObservation:
    def update(self, **_):
        pass


class Tracer:
    """Thin wrapper so the agent code is identical with and without Langfuse."""

    def __init__(self, client=None):
        self.client = client

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @contextmanager
    def observe(self, as_type: str, name: str, **kwargs):
        if not self.client:
            yield _NullObservation()
            return
        with self.client.start_as_current_observation(as_type=as_type, name=name, **kwargs) as obs:
            yield obs

    def trace_id(self) -> str | None:
        return self.client.get_current_trace_id() if self.client else None

    def flush(self) -> None:
        if self.client:
            self.client.flush()


def from_env() -> Tracer:
    """Real Langfuse client if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, else disabled."""
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse import Langfuse

        return Tracer(Langfuse())
    return Tracer(None)


def require(tracer: Tracer, allow_untraced: bool) -> None:
    if not tracer.enabled and not allow_untraced:
        raise UntracedRunError(
            "Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Agent runs must be traced; pass --allow-untraced to run anyway."
        )
