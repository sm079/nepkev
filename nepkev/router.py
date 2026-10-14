"""Serving-time routing: one customer message -> calibrated issue probabilities and a clarification probability.

The request the model sees is the evaluation export's (nepkev.export_kev with train=False): the serving question wording
(variant 0) and the taxonomy's canonical candidate order, so served predictions match the evaluated ones. Inference is
upstream Kev's: `kev.api.to_record` builds the internal record, `kev.model.admit` encodes it within the serving context,
and the checkpoint's own temperature calibrates the probabilities.

    router = KevRouter("sm079/nepkev", revision="8daf1b3...")
    router.load()                                   # downloads and loads once; minutes on a cold cache
    router.route("OTP aaudaina login garna sakdina", "wallet").intent      # 'login_otp'

Kev and torch are imported only by KevRouter.load, so code that only builds requests or uses another Router (the API
tests' stub) does not need the `kev` extra.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from .export_kev import CLAR_QID, ISSUE_QID
from .taxonomy import Taxonomy, load_taxonomy

DTYPES = ("fp32", "bf16")


@dataclass(frozen=True)
class Routing:
    """One routed message. `issue` holds calibrated probabilities over the domain's categories in canonical order."""
    domain: str
    issue: dict[str, float]
    clarification: float

    @property
    def intent(self) -> str:
        return max(self.issue, key=self.issue.get)

    @property
    def confidence(self) -> float:
        return self.issue[self.intent]

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.issue.items(), key=lambda kv: -kv[1])


class Router(Protocol):
    """What the API needs from a router. `version` identifies the weights (it is logged with every request)."""
    version: str

    def route(self, message: str, domain: str) -> Routing: ...


def serving_request(message: str, domain: str, tax: Taxonomy) -> dict:
    """The Kev System One request for one message: the evaluation export of nepkev.export_kev without labels."""
    return {"state": message, "questions": {
        ISSUE_QID: {"type": "choice", "instructions": tax.issue_instructions(domain), "criteria": tax.criteria(domain)},
        CLAR_QID: {"type": "noul", "instructions": tax.clarification_instructions(domain),
                   "criteria": dict(tax.questions.clarification.criteria)},
    }}


class KevRouter:
    """A fine-tuned Kev checkpoint (a local run directory or a Hub repo id) loaded once and shared by all requests.

    Forward passes are serialised: on CPU one pass already uses every core, so concurrent passes only add contention.
    """

    def __init__(self, run: str, revision: str | None = None, device: str = "cpu", dtype: str = "fp32",
                 threads: int | None = None, tax: Taxonomy | None = None):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        self.run, self.revision, self.device, self.dtype, self.threads = run, revision, device, dtype, threads
        self.tax = tax or load_taxonomy()
        self.version = f"{run}@{revision[:7]}" if revision else run
        self._lock = threading.Lock()
        self._model = self._tok = None

    def load(self) -> None:
        import torch
        from kev.checkpoint import Checkpoint, LoadOptions, is_hub_id

        if self.threads:
            torch.set_num_threads(self.threads)
        ref = f"{self.run}@{self.revision}" if self.revision and is_hub_id(self.run) else self.run
        opts = LoadOptions(dtype=torch.bfloat16 if self.dtype == "bf16" else None)
        self._tok, self._model = Checkpoint(ref).load(self.device, opts)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def route(self, message: str, domain: str) -> Routing:
        if not self.loaded:
            raise RuntimeError("KevRouter.route called before load()")
        import torch
        from kev.api import SystemOneRequest, to_record
        from kev.model import admit

        rec, meta = to_record(SystemOneRequest.model_validate(serving_request(message, domain, self.tax)))
        with self._lock, torch.inference_mode():
            probs = self._model.probs(admit(self._model, self._tok, rec))
        p = {m["id"]: dict(zip(m["keys"], ps.tolist())) for m, ps in zip(meta, probs)}
        return Routing(domain=domain, issue=p[ISSUE_QID], clarification=p[CLAR_QID]["true"])
