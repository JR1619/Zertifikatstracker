"""User-managed certificate store.

Statt fragiles Scraping (DB X-markets blockt Bots, Stuttgart-API existiert
nicht, Frankfurt verlangt signierte Anti-Bot-Header) pflegt der Nutzer seine
Beobachtungsliste selbst — einmal eingetragen sind die Termsheet-Daten stabil
bis zur Endfaelligkeit. Aktuelle Bid/Ask kann der Nutzer optional pflegen.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from zerttracker.models import CertificateType, ExpressCertificate, ObservationDate

DEFAULT_STORE_PATH = Path(__file__).resolve().parents[2].parent / "data" / "user_certificates.json"


def _serialize(cert: ExpressCertificate) -> dict[str, Any]:
    d = cert.model_dump(mode="json")
    return d


def _deserialize(d: dict[str, Any]) -> ExpressCertificate:
    obs = [ObservationDate(**o) for o in d.get("observations", [])]
    payload = {**d, "observations": obs}
    if isinstance(payload.get("cert_type"), str):
        payload["cert_type"] = CertificateType(payload["cert_type"])
    for k in ("issue_date", "maturity_date"):
        if isinstance(payload.get(k), str):
            payload[k] = datetime.fromisoformat(payload[k]).date()
    return ExpressCertificate(**payload)


def load_user_certificates(path: Path | None = None) -> list[ExpressCertificate]:
    p = Path(path) if path else DEFAULT_STORE_PATH
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: list[ExpressCertificate] = []
    for entry in raw:
        try:
            out.append(_deserialize(entry))
        except (ValidationError, ValueError, KeyError):
            continue
    return out


def save_user_certificates(certs: list[ExpressCertificate], path: Path | None = None) -> Path:
    p = Path(path) if path else DEFAULT_STORE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [_serialize(c) for c in certs]
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return p


def upsert_user_certificate(cert: ExpressCertificate, path: Path | None = None) -> list[ExpressCertificate]:
    certs = load_user_certificates(path)
    certs = [c for c in certs if c.isin != cert.isin]
    certs.append(cert)
    save_user_certificates(certs, path)
    return certs


def delete_user_certificate(isin: str, path: Path | None = None) -> list[ExpressCertificate]:
    certs = [c for c in load_user_certificates(path) if c.isin != isin]
    save_user_certificates(certs, path)
    return certs


def _json_default(o: Any) -> Any:
    if isinstance(o, date):
        return o.isoformat()
    raise TypeError(f"Cannot serialize {type(o).__name__}")
