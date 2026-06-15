"""Microlab Diagnostic — Liver Elastography Report Generator (Backend)."""

import csv
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Any, Dict, List

import requests
from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import Response as FastResponse, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from weasyprint import HTML as WeasyHTML

import clinical_rules

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("microlab")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

STORAGE_URL = "https://integrations.emergentagent.com/objstore/api/v1/storage"
EMERGENT_KEY = os.environ.get("EMERGENT_LLM_KEY")
APP_NAME = os.environ.get("APP_NAME", "microlab-elastography")
LOCAL_UPLOAD_DIR = Path(os.environ.get("LOCAL_UPLOAD_DIR", ROOT_DIR / "uploads"))
_storage_key: Optional[str] = None


def _safe_local_path(path: str) -> Path:
    """Resolve an app storage path into LOCAL_UPLOAD_DIR safely."""
    clean = path.strip().lstrip("/")
    target = (LOCAL_UPLOAD_DIR / clean).resolve()
    base = LOCAL_UPLOAD_DIR.resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="Invalid storage path")
    return target


def init_storage() -> str:
    """Initialise object storage.

    In Emergent, EMERGENT_LLM_KEY is used with the hosted object store.
    Outside Emergent, the app falls back to a local uploads/ directory so logo
    and signature upload continue working during clinic/VPS development.
    """
    global _storage_key
    if _storage_key:
        return _storage_key
    if not EMERGENT_KEY:
        LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        _storage_key = "local"
        return _storage_key
    resp = requests.post(
        f"{STORAGE_URL}/init", json={"emergent_key": EMERGENT_KEY}, timeout=30
    )
    resp.raise_for_status()
    _storage_key = resp.json()["storage_key"]
    return _storage_key


def put_object(path: str, data: bytes, content_type: str) -> dict:
    key = init_storage()
    if key == "local":
        target = _safe_local_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        meta = target.with_suffix(target.suffix + ".content-type")
        meta.write_text(content_type, encoding="utf-8")
        return {"path": path, "size": len(data)}
    resp = requests.put(
        f"{STORAGE_URL}/objects/{path}",
        headers={"X-Storage-Key": key, "Content-Type": content_type},
        data=data,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_object(path: str) -> tuple:
    key = init_storage()
    if key == "local":
        target = _safe_local_path(path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")
        meta = target.with_suffix(target.suffix + ".content-type")
        content_type = meta.read_text(encoding="utf-8") if meta.exists() else "application/octet-stream"
        return target.read_bytes(), content_type
    resp = requests.get(
        f"{STORAGE_URL}/objects/{path}",
        headers={"X-Storage-Key": key},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type", "application/octet-stream")


DEFAULT_CUTOFFS = {
    "fibrosis_kpa": {"F0_F1_max": 7.0, "F2_max": 9.5, "F3_max": 12.5},
    "steatosis_cap": {"S0_max": 238, "S1_max": 260, "S2_max": 291, "significant_threshold": 275},
    "reliability": {
        "iqr_median_max": 0.30,
        "success_rate_min": 60,
        "valid_measurements_min": 10,
    },
}

DEFAULT_SETTINGS_TEMPLATE = {
    "clinic_name": "Microlab Diagnostic",
    "clinic_address": "",
    "clinic_phone": "",
    "clinic_email": "",
    "logo_path": "",
    "doctor_name": "",
    "doctor_credentials": "",
    "doctor_registration": "",
    "signature_path": "",
    "disclaimer": (
        "This report is generated based on transient elastography measurements. "
        "Results should be interpreted in conjunction with clinical findings, "
        "biochemical parameters, and imaging studies. This examination is not a "
        "substitute for liver biopsy where indicated."
    ),
    "cutoffs": DEFAULT_CUTOFFS,
    "signatories": [],
    "report_templates": [],
    "active_template_id": "",
}


class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: str


class Patient(BaseModel):
    name: str = ""
    age: Optional[int] = None
    sex: str = ""
    patient_id: str = ""
    referred_by: str = ""
    date_of_exam: str = ""
    contact: str = ""
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None


class Signatory(BaseModel):
    id: str
    name: str = ""
    credentials: str = ""
    registration: str = ""
    signature_path: str = ""
    is_default: bool = False


class Clinical(BaseModel):
    indication: str = ""
    etiology: List[str] = []  # legacy multi-select kept for backward compat
    primary_etiology: str = ""  # one of: nafld | hbv | hcv | alcohol | cholestatic | mixed | ""
    etiology_other: str = ""
    exam_technique: str = "Transient Elastography (VCTE)"
    probe: str = "M"
    fasting_status: str = ""
    fasting_hours: Optional[float] = None
    clinical_notes: str = ""
    # Risk factors / confounders (booleans)
    diabetes: bool = False
    obesity: bool = False
    alcohol_use: str = ""  # none | occasional | moderate | heavy | former
    hepatic_congestion: bool = False
    acute_hepatitis: bool = False
    cholestasis_obstruction: bool = False
    focal_lesion: bool = False
    ascites: bool = False
    post_prandial: bool = False


class Labs(BaseModel):
    ast: Optional[float] = None
    alt: Optional[float] = None
    alp: Optional[float] = None
    bilirubin: Optional[float] = None
    platelet_count: Optional[float] = None
    albumin: Optional[float] = None
    inr: Optional[float] = None


class Measurements(BaseModel):
    kpa_median: Optional[float] = None
    iqr: Optional[float] = None
    iqr_median_ratio: Optional[float] = None
    valid_measurements: Optional[int] = None
    total_attempts: Optional[int] = None
    success_rate: Optional[float] = None
    cap_median: Optional[float] = None
    cap_iqr: Optional[float] = None


class ReportIn(BaseModel):
    patient: Patient = Field(default_factory=Patient)
    clinical: Clinical = Field(default_factory=Clinical)
    labs: Labs = Field(default_factory=Labs)
    measurements: Measurements = Field(default_factory=Measurements)
    operator_notes: str = ""
    final_impression: str = ""
    machine_impression: str = ""
    fibrosis_suggestion: str = ""
    steatosis_suggestion: str = ""
    reliability_flag: str = ""
    portal_ht_note: str = ""
    limitations: List[str] = []
    signatory_id: str = ""
    template_id: str = ""


class ReportOut(ReportIn):
    id: str
    user_id: str
    created_at: str
    updated_at: str


class SettingsIn(BaseModel):
    clinic_name: Optional[str] = None
    clinic_address: Optional[str] = None
    clinic_phone: Optional[str] = None
    clinic_email: Optional[str] = None
    logo_path: Optional[str] = None
    doctor_name: Optional[str] = None
    doctor_credentials: Optional[str] = None
    doctor_registration: Optional[str] = None
    signature_path: Optional[str] = None
    disclaimer: Optional[str] = None
    cutoffs: Optional[Dict[str, Any]] = None
    signatories: Optional[List[Dict[str, Any]]] = None
    report_templates: Optional[List[Dict[str, Any]]] = None
    active_template_id: Optional[str] = None


app = FastAPI(title="Microlab Elastography Report API")
api_router = APIRouter(prefix="/api")


@app.on_event("startup")
async def startup() -> None:
    try:
        init_storage()
        logger.info("Object storage initialized.")
    except Exception as exc:
        logger.error("Object storage init failed: %s", exc)


@app.on_event("shutdown")
async def shutdown_db_client() -> None:
    client.close()


async def get_current_user(
    request: Request,
    session_token: Optional[str] = Cookie(default=None),
    authorization: Optional[str] = Header(default=None),
) -> User:
    token = session_token
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session_doc = await db.user_sessions.find_one(
        {"session_token": token}, {"_id": 0}
    )
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")

    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")

    user_doc = await db.users.find_one(
        {"user_id": session_doc["user_id"]}, {"_id": 0}
    )
    if not user_doc:
        raise HTTPException(status_code=401, detail="User not found")
    return User(**user_doc)


@api_router.post("/auth/session")
async def auth_session(payload: Dict[str, str], response: Response) -> Dict[str, Any]:
    """Exchange Emergent session_id for a persistent session_token."""
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    try:
        r = requests.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": session_id},
            timeout=15,
        )
        r.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("session-data failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid session_id")

    data = r.json()
    email = data["email"]
    now = datetime.now(timezone.utc)

    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        user_id = existing["user_id"]
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": data.get("name", existing["name"]),
                      "picture": data.get("picture", existing.get("picture"))}},
        )
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one(
            {
                "user_id": user_id,
                "email": email,
                "name": data.get("name", ""),
                "picture": data.get("picture", ""),
                "created_at": now.isoformat(),
            }
        )
        await db.settings.insert_one(
            {"user_id": user_id, **DEFAULT_SETTINGS_TEMPLATE}
        )

    session_token = data["session_token"]
    expires_at = now + timedelta(days=7)
    await db.user_sessions.insert_one(
        {
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": expires_at.isoformat(),
            "created_at": now.isoformat(),
        }
    )

    response.set_cookie(
        key="session_token",
        value=session_token,
        max_age=7 * 24 * 60 * 60,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
    )

    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return {"user": user_doc, "session_token": session_token}


@api_router.get("/auth/me")
async def auth_me(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return user.model_dump()


@api_router.post("/auth/logout")
async def auth_logout(
    response: Response,
    session_token: Optional[str] = Cookie(default=None),
) -> Dict[str, str]:
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    response.delete_cookie("session_token", path="/")
    return {"status": "ok"}


async def _get_or_create_settings(user_id: str) -> Dict[str, Any]:
    doc = await db.settings.find_one({"user_id": user_id}, {"_id": 0})
    if doc:
        merged = {**DEFAULT_SETTINGS_TEMPLATE, **doc}
        return merged
    new = {"user_id": user_id, **DEFAULT_SETTINGS_TEMPLATE}
    await db.settings.insert_one(new)
    new.pop("_id", None)
    return new


@api_router.get("/settings")
async def get_settings(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return await _get_or_create_settings(user.user_id)


@api_router.put("/settings")
async def update_settings(
    payload: SettingsIn, user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    update_data = {k: v for k, v in payload.model_dump().items() if v is not None}
    await db.settings.update_one(
        {"user_id": user.user_id},
        {"$set": update_data},
        upsert=True,
    )
    return await _get_or_create_settings(user.user_id)


@api_router.post("/settings/reset-cutoffs")
async def reset_cutoffs(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    await db.settings.update_one(
        {"user_id": user.user_id},
        {"$set": {"cutoffs": DEFAULT_CUTOFFS}},
        upsert=True,
    )
    return await _get_or_create_settings(user.user_id)


# ---------------------------------------------------------------------------
# Audit log helpers
# ---------------------------------------------------------------------------
AUDIT_IGNORE_KEYS = {"updated_at", "created_at", "user_id", "id"}
AUDIT_MAX_PER_REPORT = int(os.environ.get("AUDIT_MAX_PER_REPORT", "100"))


def _diff_docs(old: Any, new: Any, path: str = "") -> List[Dict[str, Any]]:
    """Recursive diff.

    Dicts are walked. Lists are treated set-wise: emits {kind: 'list',
    added: [...], removed: [...]}. Scalars compared directly with empty/null
    equivalence so no-op edits don't pollute the audit trail.
    """
    if isinstance(old, dict) or isinstance(new, dict):
        old_d = old or {}
        new_d = new or {}
        if not isinstance(old_d, dict):
            old_d = {}
        if not isinstance(new_d, dict):
            new_d = {}
        keys = (set(old_d.keys()) | set(new_d.keys())) - AUDIT_IGNORE_KEYS
        changes: List[Dict[str, Any]] = []
        for k in keys:
            changes.extend(
                _diff_docs(old_d.get(k), new_d.get(k), f"{path}.{k}" if path else k)
            )
        return changes
    if isinstance(old, list) or isinstance(new, list):
        old_l = old if isinstance(old, list) else []
        new_l = new if isinstance(new, list) else []
        # Only support set-diff for hashable items.
        try:
            old_s = set(old_l)
            new_s = set(new_l)
        except TypeError:
            if old_l != new_l:
                return [{"path": path, "kind": "list", "old": old_l, "new": new_l}]
            return []
        added = sorted(new_s - old_s)
        removed = sorted(old_s - new_s)
        if not added and not removed:
            return []
        return [{"path": path, "kind": "list", "added": added, "removed": removed}]
    # Suppress no-op diffs between None / "" / 0 sentinels
    if old in (None, "") and new in (None, ""):
        return []
    if old != new:
        return [{"path": path, "old": old, "new": new}]
    return []


async def _write_audit(
    user: User,
    report_id: str,
    action: str,
    changes: Optional[List[Dict[str, Any]]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    entry = {
        "id": f"aud_{uuid.uuid4().hex[:12]}",
        "user_id": user.user_id,
        "actor_name": user.name,
        "actor_email": user.email,
        "report_id": report_id,
        "action": action,
        "changes": changes or [],
        "snapshot": snapshot,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await db.audit_log.insert_one(entry)
        # Trim to retention cap
        total = await db.audit_log.count_documents(
            {"report_id": report_id, "user_id": user.user_id}
        )
        if total > AUDIT_MAX_PER_REPORT:
            overflow = total - AUDIT_MAX_PER_REPORT
            old_cursor = (
                db.audit_log.find(
                    {"report_id": report_id, "user_id": user.user_id}, {"id": 1, "_id": 0}
                )
                .sort("created_at", 1)
                .limit(overflow)
            )
            ids_to_drop = [doc["id"] async for doc in old_cursor]
            if ids_to_drop:
                await db.audit_log.delete_many({"id": {"$in": ids_to_drop}})
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit write failed: %s", exc)


@api_router.get("/audit")
async def list_audit(
    report_id: str = Query(default=""),
    limit: int = Query(default=200, le=1000),
    user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {"user_id": user.user_id}
    if report_id:
        query["report_id"] = report_id
    docs = (
        await db.audit_log.find(query, {"_id": 0})
        .sort("created_at", -1)
        .to_list(limit)
    )
    return docs


@api_router.post("/reports", response_model=ReportOut)
async def create_report(
    payload: ReportIn, user: User = Depends(get_current_user)
) -> ReportOut:
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "id": f"rpt_{uuid.uuid4().hex[:12]}",
        "user_id": user.user_id,
        "created_at": now,
        "updated_at": now,
        **payload.model_dump(),
    }
    await db.reports.insert_one(doc)
    doc.pop("_id", None)
    await _write_audit(user, doc["id"], "create", changes=[], snapshot=None)
    return ReportOut(**doc)


@api_router.get("/reports")
async def list_reports(
    q: str = Query(default=""),
    limit: int = Query(default=100, le=500),
    user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {"user_id": user.user_id}
    if q:
        query["$or"] = [
            {"patient.name": {"$regex": q, "$options": "i"}},
            {"patient.patient_id": {"$regex": q, "$options": "i"}},
            {"patient.referred_by": {"$regex": q, "$options": "i"}},
        ]
    docs = (
        await db.reports.find(query, {"_id": 0})
        .sort("created_at", -1)
        .to_list(limit)
    )
    return docs


@api_router.get("/reports/{report_id}", response_model=ReportOut)
async def get_report(
    report_id: str, user: User = Depends(get_current_user)
) -> ReportOut:
    doc = await db.reports.find_one(
        {"id": report_id, "user_id": user.user_id}, {"_id": 0}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportOut(**doc)


@api_router.put("/reports/{report_id}", response_model=ReportOut)
async def update_report(
    report_id: str,
    payload: ReportIn,
    user: User = Depends(get_current_user),
) -> ReportOut:
    existing = await db.reports.find_one(
        {"id": report_id, "user_id": user.user_id}, {"_id": 0}
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Report not found")
    update = {
        **payload.model_dump(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.reports.update_one(
        {"id": report_id, "user_id": user.user_id}, {"$set": update}
    )
    doc = await db.reports.find_one(
        {"id": report_id, "user_id": user.user_id}, {"_id": 0}
    )
    changes = _diff_docs(existing, doc)
    if changes:
        await _write_audit(user, report_id, "update", changes=changes)
    return ReportOut(**doc)


@api_router.delete("/reports/{report_id}")
async def delete_report(
    report_id: str, user: User = Depends(get_current_user)
) -> Dict[str, str]:
    existing = await db.reports.find_one(
        {"id": report_id, "user_id": user.user_id}, {"_id": 0}
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Report not found")
    await db.reports.delete_one({"id": report_id, "user_id": user.user_id})
    await _write_audit(
        user, report_id, "delete", changes=[], snapshot=existing
    )
    return {"status": "deleted"}


@api_router.get("/stats/summary")
async def stats_summary(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    total = await db.reports.count_documents({"user_id": user.user_id})
    last_30 = await db.reports.count_documents(
        {
            "user_id": user.user_id,
            "created_at": {
                "$gte": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            },
        }
    )

    fibrosis: Dict[str, int] = {}
    steatosis: Dict[str, int] = {}
    reliability = {"Reliable": 0, "Unreliable": 0, "Unknown": 0}
    async for r in db.reports.find({"user_id": user.user_id}, {"_id": 0}):
        f = r.get("fibrosis_suggestion") or "—"
        s = r.get("steatosis_suggestion") or "—"
        rel = r.get("reliability_flag") or "Unknown"
        fibrosis[f] = fibrosis.get(f, 0) + 1
        steatosis[s] = steatosis.get(s, 0) + 1
        if rel in reliability:
            reliability[rel] += 1
        else:
            reliability["Unknown"] += 1
    return {
        "total_reports": total,
        "reports_last_30_days": last_30,
        "fibrosis_distribution": fibrosis,
        "steatosis_distribution": steatosis,
        "reliability_distribution": reliability,
    }


@api_router.get("/reports/by-mrn/{mrn}")
async def reports_by_mrn(
    mrn: str, user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """All reports for a given patient MRN — used for trend timeline."""
    docs = (
        await db.reports.find(
            {"user_id": user.user_id, "patient.patient_id": mrn}, {"_id": 0}
        )
        .sort("patient.date_of_exam", 1)
        .to_list(500)
    )
    return docs


@api_router.get("/reports/export/csv")
async def export_csv(
    user: User = Depends(get_current_user),
    q: str = Query(default=""),
) -> StreamingResponse:
    query: Dict[str, Any] = {"user_id": user.user_id}
    if q:
        query["$or"] = [
            {"patient.name": {"$regex": q, "$options": "i"}},
            {"patient.patient_id": {"$regex": q, "$options": "i"}},
        ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "report_id", "created_at", "date_of_exam",
        "patient_name", "patient_id", "age", "sex",
        "weight_kg", "height_cm", "bmi",
        "referred_by", "indication", "etiology",
        "technique", "probe", "fasting_status", "fasting_hours",
        "kpa_median", "iqr", "iqr_median_ratio",
        "valid_measurements", "total_attempts", "success_rate",
        "cap_median", "cap_iqr",
        "fibrosis_suggestion", "steatosis_suggestion", "reliability_flag",
        "final_impression", "operator_notes",
    ])
    async for r in db.reports.find(query, {"_id": 0}).sort("created_at", -1):
        p = r.get("patient", {}) or {}
        c = r.get("clinical", {}) or {}
        m = r.get("measurements", {}) or {}
        bmi = ""
        try:
            if p.get("weight_kg") and p.get("height_cm"):
                h_m = float(p["height_cm"]) / 100
                if h_m > 0:
                    bmi = round(float(p["weight_kg"]) / (h_m * h_m), 1)
        except (TypeError, ValueError):
            bmi = ""
        writer.writerow([
            r.get("id", ""),
            r.get("created_at", ""),
            p.get("date_of_exam", ""),
            p.get("name", ""),
            p.get("patient_id", ""),
            p.get("age", ""),
            p.get("sex", ""),
            p.get("weight_kg", ""),
            p.get("height_cm", ""),
            bmi,
            p.get("referred_by", ""),
            c.get("indication", ""),
            "; ".join(c.get("etiology", []) or []),
            c.get("exam_technique", ""),
            c.get("probe", ""),
            c.get("fasting_status", ""),
            c.get("fasting_hours", ""),
            m.get("kpa_median", ""),
            m.get("iqr", ""),
            m.get("iqr_median_ratio", ""),
            m.get("valid_measurements", ""),
            m.get("total_attempts", ""),
            m.get("success_rate", ""),
            m.get("cap_median", ""),
            m.get("cap_iqr", ""),
            r.get("fibrosis_suggestion", ""),
            r.get("steatosis_suggestion", ""),
            r.get("reliability_flag", ""),
            (r.get("final_impression", "") or "").replace("\n", " "),
            (r.get("operator_notes", "") or "").replace("\n", " "),
        ])
    buffer.seek(0)
    filename = f"microlab-reports-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _bmi(weight_kg: Any, height_cm: Any) -> Optional[float]:
    try:
        if weight_kg in (None, "") or height_cm in (None, ""):
            return None
        h_m = float(height_cm) / 100
        if h_m <= 0:
            return None
        return round(float(weight_kg) / (h_m * h_m), 1)
    except (TypeError, ValueError):
        return None


def _img_data_uri_from_storage(path: str) -> Optional[str]:
    """Fetch an uploaded image from object storage and inline as data URI."""
    if not path:
        return None
    try:
        data, content_type = get_object(path)
        import base64
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{content_type};base64,{b64}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image fetch failed for %s: %s", path, exc)
        return None


def _resolve_signatory(settings: Dict[str, Any], signatory_id: str) -> Dict[str, Any]:
    sigs = settings.get("signatories") or []
    if signatory_id:
        for s in sigs:
            if s.get("id") == signatory_id:
                return s
    # First default signatory
    for s in sigs:
        if s.get("is_default"):
            return s
    if sigs:
        return sigs[0]
    # Fall back to legacy fields
    return {
        "name": settings.get("doctor_name", ""),
        "credentials": settings.get("doctor_credentials", ""),
        "registration": settings.get("doctor_registration", ""),
        "signature_path": settings.get("signature_path", ""),
    }


def _resolve_template(settings: Dict[str, Any], template_id: str) -> Dict[str, Any]:
    templates = settings.get("report_templates") or []
    if template_id:
        for t in templates:
            if t.get("id") == template_id:
                return t
    active_id = settings.get("active_template_id")
    if active_id:
        for t in templates:
            if t.get("id") == active_id:
                return t
    return {"id": "", "name": "", "language": "en", "header_note": "", "impression_presets": []}


def _load_translations() -> Dict[str, Dict[str, str]]:
    """Load i18n strings from /app/backend/i18n/*.json (extracted for translator workflow)."""
    base = ROOT_DIR / "i18n"
    out: Dict[str, Dict[str, str]] = {}
    if base.exists():
        for path in base.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out[path.stem] = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("i18n load failed for %s: %s", path, exc)
    if "en" not in out:
        out["en"] = {}
    return out


PDF_TRANSLATIONS: Dict[str, Dict[str, str]] = _load_translations()


def _t(lang: str, key: str) -> str:
    return PDF_TRANSLATIONS.get(lang or "en", PDF_TRANSLATIONS["en"]).get(
        key, PDF_TRANSLATIONS["en"].get(key, key)
    )


def _render_report_html(report: Dict[str, Any], settings: Dict[str, Any]) -> str:
    p = report.get("patient", {}) or {}
    c = report.get("clinical", {}) or {}
    labs = report.get("labs", {}) or {}
    m = report.get("measurements", {}) or {}
    cutoffs = settings.get("cutoffs", DEFAULT_CUTOFFS) or DEFAULT_CUTOFFS
    sig = _resolve_signatory(settings, report.get("signatory_id", "") or "")
    template = _resolve_template(settings, report.get("template_id", "") or "")
    lang = template.get("language") or "en"
    header_note = template.get("header_note") or ""
    bmi = _bmi(p.get("weight_kg"), p.get("height_cm"))

    # Clinical rules — recompute on the backend for the PDF in case the client
    # didn't send them (or sent stale ones).
    primary_etiology = c.get("primary_etiology") or ""
    fibrosis = clinical_rules.fibrosis_from_kpa(m.get("kpa_median"), primary_etiology)
    steatosis = clinical_rules.steatosis_from_cap(m.get("cap_median"))
    quality = clinical_rules.quality_class(m)
    portal_ht = report.get("portal_ht_note") or clinical_rules.portal_hypertension_note(
        m.get("kpa_median"), labs.get("platelet_count")
    )
    limitations = report.get("limitations") or clinical_rules.confounder_warnings(c)
    etiology_label = clinical_rules.ETIOLOGY_LABELS.get(primary_etiology, "")
    machine_imp = report.get("machine_impression") or clinical_rules.build_machine_impression(
        fibrosis, steatosis, quality, etiology_label, m.get("cap_median")
    )

    def t(k):
        return _t(lang, k)

    def fmt(v, digits=1):
        try:
            if v in (None, ""):
                return "—"
            return f"{float(v):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    def fmt_int(v):
        try:
            if v in (None, ""):
                return "—"
            return str(int(float(v)))
        except (TypeError, ValueError):
            return "—"

    ratio_val = m.get("iqr_median_ratio")
    if ratio_val in (None, "") and m.get("iqr") and m.get("kpa_median"):
        try:
            ratio_val = float(m["iqr"]) / float(m["kpa_median"])
        except (TypeError, ZeroDivisionError, ValueError):
            ratio_val = None

    logo_uri = _img_data_uri_from_storage(settings.get("logo_path", ""))
    sig_uri = _img_data_uri_from_storage(sig.get("signature_path", ""))

    # Etiology text: prefer primary etiology label; fall back to legacy list/other
    if primary_etiology:
        etiology_text = etiology_label
        if c.get("etiology_other"):
            etiology_text += f" · {c['etiology_other']}"
    else:
        etiology_text = "; ".join(c.get("etiology", []) or [])
        if c.get("etiology_other"):
            etiology_text = (etiology_text + "; " if etiology_text else "") + c["etiology_other"]

    fk = cutoffs.get("fibrosis_kpa", {})
    sk = cutoffs.get("steatosis_cap", {})

    # Labs row content
    lab_defs = [
        ("ast", "AST", "U/L"), ("alt", "ALT", "U/L"), ("alp", "ALP", "U/L"),
        ("bilirubin", "Bilirubin", "mg/dL"),
        ("platelet_count", "Platelets", "×10⁹/L"),
        ("albumin", "Albumin", "g/dL"), ("inr", "INR", "ratio"),
    ]
    has_labs = any(labs.get(k) not in (None, "") for k, _, _ in lab_defs)
    labs_rows = ""
    if has_labs:
        cells = "".join(
            f'<td style="padding:3px 6px;border-bottom:1px solid #e4e4e7;"><strong>{lab}</strong> '
            f'<span class="num">{fmt(labs.get(k), 2 if k in ("bilirubin","albumin","inr") else 0)}</span> '
            f'<span style="color:#52525b">{u}</span></td>'
            for k, lab, u in lab_defs
            if labs.get(k) not in (None, "")
        )
        labs_rows = f'<h2>{t("labs")}</h2><table><tr>{cells}</tr></table>'

    # Limitations
    limitations_html = ""
    if limitations:
        items = "".join(f"<li>{w}</li>" for w in limitations)
        limitations_html = f'<h2>{t("limitations")}</h2><ul style="margin:4px 0 0 14px;padding:0;font-size:9.5pt;color:#27272a;">{items}</ul>'

    # Portal HT note
    portal_html = ""
    if portal_ht:
        portal_html = f'<h2>{t("portal_ht")}</h2><div style="font-size:10pt;color:#27272a;">{portal_ht}</div>'

    # Clinic-ready HTML/PDF layout.
    import html as _html

    def esc(v: Any) -> str:
        return _html.escape(str(v if v not in (None, "") else "—"))

    def esc_raw(v: Any) -> str:
        return _html.escape(str(v if v not in (None, "") else ""))

    def nl2br(v: Any) -> str:
        return "<br>".join(esc_raw(v).splitlines()) if v not in (None, "") else "—"

    clinical_bits = []
    if c.get("indication"):
        clinical_bits.append(c.get("indication"))
    if etiology_text:
        clinical_bits.append(f"Etiology: {etiology_text}")
    if c.get("diabetes"):
        clinical_bits.append("Diabetes")
    if c.get("obesity"):
        clinical_bits.append("Obesity")
    if c.get("alcohol_use"):
        clinical_bits.append(f"Alcohol use: {c.get('alcohol_use')}")
    if c.get("clinical_notes"):
        clinical_bits.append(c.get("clinical_notes"))
    clinical_context = " · ".join(str(x) for x in clinical_bits) or "—"

    final_impression = report.get("final_impression", "") or machine_imp or ""
    impression_items = "".join(
        f"<li>{esc_raw(line.strip())}</li>"
        for line in final_impression.splitlines()
        if line.strip()
    )
    impression_html = f'<ol class="impression-list">{impression_items}</ol>' if impression_items else '<div class="text-block">—</div>'

    limitations_html = ""
    if limitations:
        items = "".join(f"<li>{esc_raw(w)}</li>" for w in limitations)
        limitations_html = f'<h2>{t("limitations")}</h2><ul class="paper-list">{items}</ul>'

    portal_html = ""
    if portal_ht:
        portal_html = f'<h2>{t("portal_ht")}</h2><div class="text-block">{esc_raw(portal_ht)}</div>'

    labs_html = ""
    if has_labs:
        rows = "".join(
            f'<tr><td>{lab}</td><td class="num">{fmt(labs.get(k), 2 if k in ("bilirubin","albumin","inr") else 0)} <span>{u}</span></td></tr>'
            for k, lab, u in lab_defs
            if labs.get(k) not in (None, "")
        )
        labs_html = f'<h2>{t("labs")}</h2><table class="data">{rows}</table>'

    if primary_etiology and clinical_rules.FIBROSIS_TABLES.get(primary_etiology):
        tbl = clinical_rules.FIBROSIS_TABLES[primary_etiology]
        fibrosis_cutoffs = (
            f"Fibrosis: F0–F1 < {tbl['f2'][0]} · F2 {tbl['f2'][0]}–{tbl['f2'][1]} · "
            f"F3 {tbl['f3'][0]}–{tbl['f3'][1]} · F4 ≥ {tbl['f4']} kPa"
        )
    else:
        fibrosis_cutoffs = "Fibrosis: Low ≤ 7.0 · Intermediate 7.1–13.9 · High ≥ 14.0 kPa (mixed / uncertain etiology)"
    cap_cutoffs = (
        f"CAP: S0 ≤ {clinical_rules.CAP_CUTOFFS['s0_max_inclusive']} · "
        f"S1 {clinical_rules.CAP_CUTOFFS['s0_max_inclusive'] + 1}–{clinical_rules.CAP_CUTOFFS['s1_max_inclusive']} · "
        f"S2 {clinical_rules.CAP_CUTOFFS['s1_max_inclusive'] + 1}–{clinical_rules.CAP_CUTOFFS['s2_max_inclusive']} · "
        f"S3 ≥ {clinical_rules.CAP_CUTOFFS['s2_max_inclusive'] + 1} dB/m"
    )

    quality_flag = quality.get("flag") or report.get("reliability_flag", "—") or "—"
    quality_class_name = "success" if quality_flag == "Reliable" else ("critical" if quality_flag == "Suboptimal" else "warning" if quality_flag == "Acceptable with caution" else "neutral")
    fibrosis_stage = fibrosis.get("stage") or report.get("fibrosis_suggestion", "—") or "—"
    steatosis_stage = steatosis.get("stage") or report.get("steatosis_suggestion", "—") or "—"
    logo_html = f'<img src="{logo_uri}" alt="logo" class="logo">' if logo_uri else '<span class="logo-fallback">M</span>'
    sig_html = f'<img src="{sig_uri}" alt="signature" class="signature-img">' if sig_uri else '<div style="height:42px"></div>'

    return f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>Liver Elastography Report</title>
<style>
  @page {{ size: A4; margin: 13mm 13mm 11mm 13mm; }}
  body {{ font-family: 'FreeSans', 'Helvetica', 'Arial', sans-serif; color:#09090b; font-size:10pt; margin:0; }}
  .header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; border-bottom:2px solid #09090b; padding-bottom:7px; }}
  .brand {{ display:flex; align-items:center; gap:10px; min-width:0; }}
  .logo, .logo-fallback {{ width:46px; height:46px; object-fit:contain; flex:0 0 auto; }}
  .logo-fallback {{ background:#09090b; color:white; display:inline-block; text-align:center; line-height:46px; font-weight:800; font-size:18pt; }}
  .clinic-name {{ font-size:14pt; font-weight:800; letter-spacing:-0.01em; }}
  .clinic-addr, .report-meta, .report-subtitle {{ font-size:8pt; color:#52525b; line-height:1.35; }}
  .title-block {{ text-align:right; width:270px; }}
  .report-title {{ font-size:10.5pt; font-weight:800; text-transform:uppercase; letter-spacing:0.08em; }}
  .patient-band {{ display:grid; grid-template-columns:1.4fr .8fr 1fr 1.1fr; gap:6px 18px; margin-top:9px; padding:7px 8px; border:1px solid #d4d4d8; background:#fafafa; }}
  .lbl {{ font-size:7.2pt; color:#52525b; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
  .val {{ font-size:9.5pt; font-weight:600; line-height:1.25; }}
  .results {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin-top:9px; }}
  .card {{ border:1px solid #d4d4d8; border-left:3px solid #71717a; padding:6px 7px; min-height:55px; }}
  .card.success {{ border-left-color:#059669; }} .card.warning {{ border-left-color:#92400e; }} .card.critical {{ border-left-color:#dc2626; }}
  .card-label {{ font-size:7.3pt; color:#52525b; font-weight:800; text-transform:uppercase; letter-spacing:.1em; }}
  .card-value {{ font-size:13pt; font-weight:800; margin-top:1px; }} .card-value span {{ font-size:8pt; color:#52525b; font-weight:500; }}
  .card-sub {{ font-size:8pt; color:#52525b; line-height:1.25; margin-top:1px; }}
  h2 {{ font-size:8.5pt; font-weight:800; text-transform:uppercase; letter-spacing:.1em; border-bottom:1px solid #09090b; padding-bottom:2px; margin:9px 0 5px; }}
  .text-block {{ font-size:9.4pt; color:#27272a; line-height:1.35; white-space:pre-wrap; }}
  table {{ width:100%; border-collapse:collapse; font-size:9.4pt; }} table.data {{ border:1px solid #d4d4d8; }}
  table.data td {{ padding:2.5px 6px; border-bottom:1px solid #e4e4e7; vertical-align:top; }}
  table.data .num {{ text-align:right; font-family:'Courier', monospace; font-weight:700; }} table.data .num span {{ color:#52525b; font-weight:400; }}
  td span, td em {{ color:#52525b; font-weight:400; }}
  .success {{ color:#047857; }} .warning {{ color:#92400e; }} .critical {{ color:#dc2626; }}
  .paper-list {{ margin:4px 0 0 14px; padding:0; font-size:9.2pt; color:#27272a; line-height:1.3; }}
  .impression-list {{ margin:0 0 0 17px; padding:0; font-size:10pt; font-weight:600; line-height:1.45; }}
  .footer {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px; margin-top:14px; }}
  .cutoffs {{ font-size:7.4pt; color:#52525b; max-width:66%; line-height:1.4; }} .cutoffs strong {{ color:#09090b; }}
  .signature {{ text-align:center; min-width:176px; }} .signature-img {{ max-height:42px; max-width:165px; object-fit:contain; }}
  .sig-line {{ border-top:1px solid #09090b; padding-top:2px; margin-top:3px; }} .sig-name {{ font-size:9.5pt; font-weight:800; }} .sig-cred {{ font-size:8pt; color:#52525b; }}
  .disclaimer {{ font-size:7.4pt; color:#52525b; border-top:1px solid #d4d4d8; padding-top:5px; margin-top:10px; line-height:1.35; }}
</style>
</head><body>
  <div class="header">
    <div class="brand">{logo_html}<div><div class="clinic-name">{esc_raw(settings.get('clinic_name') or 'Microlab Diagnostic')}</div><div class="clinic-addr">{esc_raw(' · '.join(x for x in [settings.get('clinic_address',''), settings.get('clinic_phone',''), settings.get('clinic_email','')] if x) or 'Clinic address')}</div></div></div>
    <div class="title-block"><div class="report-title">Liver Elastography Report</div><div class="report-subtitle">Transient elastography / CAP assessment</div><div class="report-meta">Report No: {esc_raw((report.get('id') or p.get('patient_id') or 'Draft')).upper()} · Date: {esc(p.get('date_of_exam'))}</div>{f'<div class="report-meta" style="font-weight:600;color:#27272a">{esc_raw(header_note)}</div>' if header_note else ''}</div>
  </div>

  <div class="patient-band">
    <div><div class="lbl">Patient name</div><div class="val">{esc(p.get('name'))}</div></div>
    <div><div class="lbl">Age / Sex</div><div class="val">{esc(p.get('age'))}{' yrs' if p.get('age') else ''} · {esc(p.get('sex'))}</div></div>
    <div><div class="lbl">Patient ID / MRN</div><div class="val">{esc(p.get('patient_id'))}</div></div>
    <div><div class="lbl">Referred by</div><div class="val">{esc(p.get('referred_by'))}</div></div>
    <div><div class="lbl">Weight / Height / BMI</div><div class="val">{fmt(p.get('weight_kg'),1)} kg · {fmt(p.get('height_cm'),1)} cm · BMI {fmt(bmi,1) if bmi is not None else '—'}</div></div>
    <div><div class="lbl">Technique / Probe</div><div class="val">{esc_raw(c.get('exam_technique') or '—')} · Probe {esc(c.get('probe'))}</div></div>
  </div>

  <div class="results">
    <div class="card {'critical' if fibrosis_stage in ('F4','High') else 'success' if fibrosis_stage != '—' else ''}"><div class="card-label">Liver stiffness</div><div class="card-value">{fmt(m.get('kpa_median'),1)} <span>kPa</span></div><div class="card-sub">{esc_raw(fibrosis_stage)}{(' · ' + esc_raw(fibrosis.get('label'))) if fibrosis.get('label') else ''}</div></div>
    <div class="card {'warning' if steatosis.get('significant') else 'success' if steatosis_stage != '—' else ''}"><div class="card-label">CAP</div><div class="card-value">{fmt(m.get('cap_median'),0)} <span>dB/m</span></div><div class="card-sub">{esc_raw(steatosis_stage)}{(' · ' + esc_raw(steatosis.get('label'))) if steatosis.get('label') else ''}</div></div>
    <div class="card {quality_class_name}"><div class="card-label">Acquisition quality</div><div class="card-value">{esc_raw(quality_flag)}</div><div class="card-sub">{esc_raw(quality.get('reason') or 'Enter values to assess quality')}</div></div>
  </div>

  <h2>Clinical context</h2><div class="text-block">{esc_raw(clinical_context)}</div>

  <h2>{t('measurements')}</h2>
  <table class="data">
    <tr><td style="width:55%">{t('stiffness_median')}</td><td class="num">{fmt(m.get('kpa_median'),1)} <span>kPa</span></td></tr>
    <tr><td>{t('iqr')}</td><td class="num">{fmt(m.get('iqr'),1)} <span>kPa</span></td></tr>
    <tr><td>{t('iqr_median')}</td><td class="num">{fmt(ratio_val,2)}</td></tr>
    <tr><td>{t('valid_meas')} / total attempts</td><td class="num">{fmt_int(m.get('valid_measurements'))}{(' / ' + fmt_int(m.get('total_attempts'))) if m.get('total_attempts') not in (None,'') else ''}</td></tr>
    <tr><td>{t('success_rate')}</td><td class="num">{fmt(m.get('success_rate'),0)} <span>%</span></td></tr>
    <tr><td>{t('cap_median')}</td><td class="num">{fmt(m.get('cap_median'),0)} <span>dB/m</span></td></tr>
    <tr><td>{t('cap_iqr')}</td><td class="num">{fmt(m.get('cap_iqr'),0)} <span>dB/m</span></td></tr>
    <tr><td>Fasting status</td><td class="num">{esc_raw(c.get('fasting_status') or '—')}{(' · ' + str(c.get('fasting_hours')) + ' h') if c.get('fasting_hours') not in (None,'') else ''}</td></tr>
  </table>

  <h2>{t('interpretation')}</h2>
  <table class="data">
    <tr><td style="width:32%">{t('fibrosis')}</td><td><strong>{esc_raw(fibrosis_stage)}</strong><span>{(' · ' + esc_raw(fibrosis.get('label'))) if fibrosis.get('label') else ''}</span>{(' <em>· ' + esc_raw(etiology_label) + ' reference</em>') if etiology_label else ''}</td></tr>
    <tr><td>{t('steatosis')}</td><td><strong>{esc_raw(steatosis_stage)}</strong><span>{(' · ' + esc_raw(steatosis.get('label'))) if steatosis.get('label') else ''}</span>{'<em> · significant steatosis likely</em>' if steatosis.get('significant') else ''}</td></tr>
    <tr><td>{t('reliability')}</td><td><strong class="{quality_class_name}">{esc_raw(quality_flag)}</strong><span>{(' · ' + esc_raw(quality.get('reason'))) if quality.get('reason') else ''}</span></td></tr>
  </table>

  {labs_html}
  {portal_html}
  {limitations_html}

  <h2>{t('impression')}</h2>
  {impression_html}

  {('<h2>' + t('operator_notes') + '</h2><div class="text-block">' + nl2br(report.get('operator_notes')) + '</div>') if report.get('operator_notes') else ''}

  <div class="footer">
    <div class="cutoffs"><strong>{t('cutoffs_used')}</strong><br>{esc_raw(fibrosis_cutoffs)}<br>{esc_raw(cap_cutoffs)}</div>
    <div class="signature">{sig_html}<div class="sig-line"><div class="sig-name">{esc(sig.get('name'))}</div><div class="sig-cred">{esc_raw(' · '.join(x for x in [sig.get('credentials',''), ('Reg ' + sig.get('registration','')) if sig.get('registration') else ''] if x))}</div></div></div>
  </div>

  <div class="disclaimer">{esc_raw(settings.get('disclaimer') or 'This report is generated from elastography measurements and should be interpreted with clinical findings, biochemical parameters and other imaging studies. Repeat acquisition may be required if quality is suboptimal or confounders are present.')}</div>
</body></html>
""".strip()


@api_router.get("/reports/{report_id}/pdf")
async def report_pdf(
    report_id: str, user: User = Depends(get_current_user)
) -> Response:
    doc = await db.reports.find_one(
        {"id": report_id, "user_id": user.user_id}, {"_id": 0}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")
    settings = await _get_or_create_settings(user.user_id)
    html = _render_report_html(doc, settings)
    pdf_bytes = WeasyHTML(string=html).write_pdf()
    patient_name = (doc.get("patient", {}) or {}).get("name") or "report"
    safe = "".join(ch for ch in patient_name if ch.isalnum() or ch in ("-", "_")).strip("-_") or "report"
    filename = f"microlab-{safe}-{report_id[-6:]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@api_router.post("/reports/preview-pdf")
async def report_preview_pdf(
    payload: ReportIn, user: User = Depends(get_current_user)
) -> Response:
    """Render a PDF from an unsaved payload (for live download from the form)."""
    settings = await _get_or_create_settings(user.user_id)
    doc = payload.model_dump()
    html = _render_report_html(doc, settings)
    pdf_bytes = WeasyHTML(string=html).write_pdf()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="microlab-report.pdf"'},
    )


ALLOWED_KINDS = {"logo", "signature"}
ALLOWED_MIME = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/svg+xml",
}


@api_router.post("/files/upload")
async def upload_file(
    kind: str = Query(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    if kind not in ALLOWED_KINDS:
        raise HTTPException(status_code=400, detail="kind must be logo or signature")
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {content_type}")
    data = await file.read()
    if len(data) > 4 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 4MB")
    ext = (file.filename.rsplit(".", 1)[-1] if "." in (file.filename or "") else "bin").lower()
    path = f"{APP_NAME}/{kind}/{user.user_id}/{uuid.uuid4().hex}.{ext}"
    result = put_object(path, data, content_type)
    await db.files.insert_one(
        {
            "id": uuid.uuid4().hex,
            "user_id": user.user_id,
            "kind": kind,
            "storage_path": result["path"],
            "content_type": content_type,
            "size": result.get("size", len(data)),
            "is_deleted": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {"path": result["path"], "content_type": content_type}


@api_router.get("/files/raw")
async def get_file_raw(
    path: str = Query(...),
    auth: Optional[str] = Query(default=None),
    session_token: Optional[str] = Cookie(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Response:
    token = session_token
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token and auth:
        token = auth
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    session_doc = await db.user_sessions.find_one(
        {"session_token": token}, {"_id": 0}
    )
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    record = await db.files.find_one(
        {"storage_path": path, "user_id": session_doc["user_id"], "is_deleted": False},
        {"_id": 0},
    )
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    data, content_type = get_object(path)
    return FastResponse(content=data, media_type=record.get("content_type", content_type))


@api_router.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "service": "microlab-elastography"}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
