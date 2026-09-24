"""
EventBackend: pluggable strategy for finding candidate observation events.

DynamicBackend (supported)
--------------------------
Computes observation windows on-the-fly from orbital parameters via vectorised
numpy modular arithmetic.  No event table is pre-computed.  This is what
``ArielEnv`` uses.  For analysis, call ``DynamicBackend.export_events`` to
materialise a table over a time window.

    from aRieL.simulator.event_backend import DynamicBackend
    env = ArielEnv(config, targets=targets, backend=DynamicBackend(targets))
    table = env.export_events()

TableBackend (legacy)
---------------------
Works from a pre-computed event DataFrame.  Kept so older tests and
diagnostics still run; new code should use ``DynamicBackend``.

Candidate DataFrame schema
--------------------------
Both ``candidates()`` calls return a DataFrame with these columns:

    event_id               int64   – unique within a step (synthetic for Dynamic)
    target_id              object  – string target identifier
    event_type             object  – "transit" or "eclipse"
    window_start           float64 – BJD of observation window start
    window_mid             float64 – BJD of mid-point (transit/eclipse centre)
    window_end             float64 – BJD of observation window end
    duration               float64 – event duration in seconds
    duration_days          float64 – event duration in days
    tier_goal              int64   – maximum tier achievable for this target
    base_science_value     float64 – static science priority in [0, 1]
    visibility_valid       bool    – True if geometrically observable (MVP: always True)
    ephemeris_uncertainty  float64 – ±1σ timing uncertainty in days
    event_index            int64   – periodic index n (−1 for DynamicBackend)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Canonical column order — both backends always return this schema.
EVENT_COLUMNS: list[str] = [
    "event_id", "target_id", "event_type",
    "window_start", "window_mid", "window_end",
    "duration", "duration_days", "block_duration_days", "tier_goal",
    "base_science_value", "visibility_valid",
    "ephemeris_uncertainty", "event_index",
]

_EMPTY_EVENTS = pd.DataFrame(columns=EVENT_COLUMNS)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EventBackend(ABC):
    """Interface that any event-candidate provider must satisfy."""

    @abstractmethod
    def candidates(
        self, t_now: float, k: int, unstarted_only: bool = False,
    ) -> pd.DataFrame:
        """Return up to *k* upcoming events sorted by ``window_mid``.

        Parameters
        ----------
        t_now:
            Current mission time (BJD).
        k:
            Maximum number of candidates to return.
        unstarted_only:
            If True, skip observation blocks that have already opened
            (``block_start ≤ t_now``) and return the next future occurrence.

        Returns
        -------
        pd.DataFrame with columns ``EVENT_COLUMNS``, sorted nearest-first.
        May return fewer than *k* rows if events are scarce.
        """
        ...

    @abstractmethod
    def get_event(self, event_id: int) -> pd.Series:
        """Retrieve full details of a single event by its ``event_id``.

        For ``TableBackend``: O(1) indexed lookup in the pre-computed table.
        For ``DynamicBackend``: O(1) lookup in the per-step candidate cache
        (``candidates()`` must have been called in this step first).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any per-episode mutable state (e.g. sliding-window pointer)."""
        ...

    def events_for_target(
        self,
        target_id: str,
        t_now: float,
        n: int = 3,
    ) -> list[dict]:
        """Return the next *n* event dicts for a specific target starting from *t_now*.

        This is the single source of truth for future-event timing features in
        ``planet_feature_builder``.  The first returned event is always the same
        occurrence that would be executed if this target were selected right now
        (consistent with ``candidates()``).

        Parameters
        ----------
        target_id:
            Identifier of the target to look up.
        t_now:
            Current mission time (BJD).
        n:
            How many successive events to return.

        Returns
        -------
        list of dicts, each containing at minimum:
            ``window_mid``, ``block_duration_days``, ``event_type``.
        Missing events (e.g. beyond mission end) are represented as empty dicts.
        The default implementation returns an empty list; override in subclasses.
        """
        return []

    def register_event(self, event_dict: dict) -> int:
        """Register an event dict in the per-step cache and return its event_id.

        Used by ``full_set`` candidate selection to cache events from
        ``events_for_target()`` so that ``execute_observation`` can look them
        up via ``get_event(event_id)``.

        For ``TableBackend`` events are stable so no registration is needed —
        the default implementation returns the existing ``event_id`` unchanged.
        For ``DynamicBackend`` the event is written into ``_event_cache``.

        Returns -1 if registration fails (e.g. unknown target).
        """
        return int(event_dict.get("event_id", -1))


# ---------------------------------------------------------------------------
# TableBackend
# ---------------------------------------------------------------------------

class TableBackend(EventBackend):
    """Backend backed by a pre-computed events DataFrame.

    .. deprecated::
        Use :class:`DynamicBackend` instead.  ``TableBackend`` requires a
        pre-generated event table (``generate_events.py``) and does not
        populate the ``block_duration_days`` column added in Phase 1b.
        It is kept for backward-compatibility only and will be removed.

    Wraps all of the optimised access patterns previously embedded directly in
    ``MissionState`` and ``ArielEnv._candidates_topk``:

    * Binary-search sliding window avoids O(N) boolean filter.
    * Pre-indexed DataFrame for O(1) ``get_event`` lookups.
    """

    def __init__(self, events: pd.DataFrame) -> None:
        self._events = events
        self._events_idx: pd.DataFrame = (
            events.set_index("event_id")
            if len(events) and "event_id" in events.columns
            else pd.DataFrame()
        )
        self._event_wmid: np.ndarray = (
            events["window_mid"].to_numpy()
            if len(events)
            else np.array([], dtype=np.float64)
        )
        self._max_duration_days: float = (
            float(events["duration_days"].max()) if len(events) else 1.0
        )

    # ------------------------------------------------------------------
    # EventBackend interface
    # ------------------------------------------------------------------

    def candidates(
        self, t_now: float, k: int, unstarted_only: bool = False,
    ) -> pd.DataFrame:
        if len(self._events) == 0:
            return _EMPTY_EVENTS.copy()

        # Binary search: find the first event whose window_mid is in range.
        # Events before this pointer all have window_end ≤ t_now (expired).
        ptr = int(np.searchsorted(
            self._event_wmid,
            t_now - self._max_duration_days,
        ))

        # Grab a small window forward — events are already sorted by window_mid.
        scan = max(k * 10, 200)
        window = self._events.iloc[ptr: ptr + scan]

        # Exact filter on the small window.
        wend = window["window_end"].to_numpy()
        valid = wend > t_now
        if unstarted_only:
            from aRieL.data.schemas import COST_FACTOR
            if "block_duration_days" in window.columns:
                bldur = window["block_duration_days"].to_numpy(dtype=float)
            else:
                bldur = COST_FACTOR * window["duration_days"].to_numpy(dtype=float)
            block_start = window["window_mid"].to_numpy(dtype=float) - bldur / 2.0
            valid = valid & (block_start > t_now)
        valid_idx = np.where(valid)[0]

        return window.iloc[valid_idx[:k]].reset_index(drop=True)

    def get_event(self, event_id: int) -> pd.Series:
        return self._events_idx.loc[event_id]

    def events_for_target(
        self,
        target_id: str,
        t_now: float,
        n: int = 3,
    ) -> list[dict]:
        """Return the next n events for *target_id* from the pre-computed table."""
        if len(self._events) == 0:
            return []
        from aRieL.data.schemas import COST_FACTOR
        sub = self._events[self._events["target_id"] == target_id]
        # Find events whose block hasn't ended yet
        if "block_duration_days" in sub.columns:
            block_ends = sub["window_mid"] + sub["block_duration_days"] / 2.0
        else:
            block_ends = sub["window_mid"] + COST_FACTOR * sub["duration_days"] / 2.0
        sub = sub[block_ends > t_now].sort_values("window_mid")
        out = []
        for _, row in sub.head(n).iterrows():
            bd = float(row.get("block_duration_days", COST_FACTOR * row["duration_days"]))
            out.append({
                "window_mid":          float(row["window_mid"]),
                "window_start":        float(row["window_start"]),
                "window_end":          float(row["window_end"]),
                "duration_days":       float(row["duration_days"]),
                "block_duration_days": bd,
                "event_type":          str(row.get("event_type", "transit")),
            })
        return out

    def reset(self) -> None:
        pass  # Stateless: no per-episode mutable data.


# ---------------------------------------------------------------------------
# DynamicBackend
# ---------------------------------------------------------------------------

class DynamicBackend(EventBackend):
    """Backend that computes observation windows on-the-fly.

    Uses vectorised numpy modular arithmetic over all targets simultaneously
    so the per-step cost is O(N_targets) ≈ 0.05 ms for ~800 targets —
    independent of mission duration.

    Synthetic event IDs
    -------------------
    Because there is no pre-computed table, event IDs are assigned
    deterministically as::

        transit:  target_index * 2
        eclipse:  target_index * 2 + 1

    These IDs are valid only for the duration of one step; the per-step
    candidate cache is refreshed by each ``candidates()`` call.

    Limitations
    -----------
    * ``get_event()`` requires a prior ``candidates()`` call in the same step.
    * Ephemeris uncertainty is set to 0 (simplified; add a noise model later).
    * The ``target`` action space type (one event per target) is not yet
      supported; use ``topk`` with this backend.
    """

    def __init__(self, targets: pd.DataFrame) -> None:
        from aRieL.simulator.event_generator import _base_science_value
        from aRieL.data.schemas import METHOD_ECLIPSE, METHOD_EITHER, METHOD_TRANSIT, COST_FACTOR
        self._cost_factor: float = COST_FACTOR

        n = len(targets)
        self._target_ids: np.ndarray = targets["target_id"].to_numpy()
        # Reverse index: target_id → array index (for events_for_target)
        self._tid_to_idx: dict[str, int] = {
            tid: i for i, tid in enumerate(self._target_ids)
        }

        # Orbital parameters (all in days).
        self._epochs  = targets["epoch"].to_numpy(dtype=float)
        self._periods = targets["period"].to_numpy(dtype=float)

        # Transit duration.
        tr_dur_s = targets["transit_duration"].to_numpy(dtype=float)  # seconds
        self._tr_dur_s    = tr_dur_s
        self._tr_dur_days = tr_dur_s / 86400.0
        self._half_tr     = self._tr_dur_days / 2.0
        self._has_transit = np.isfinite(tr_dur_s) & (tr_dur_s > 0)

        # Eclipse duration (fall back to transit duration when absent).
        ec_dur_s = targets["eclipse_duration"].to_numpy(dtype=float)
        missing_ec = ~np.isfinite(ec_dur_s) | (ec_dur_s <= 0)
        ec_dur_s = np.where(missing_ec, tr_dur_s, ec_dur_s)
        self._ec_dur_s    = ec_dur_s
        self._ec_dur_days = ec_dur_s / 86400.0
        self._half_ec     = self._ec_dur_days / 2.0

        # Which targets observe transits vs eclipses?
        # preferred_method == "Transit"  → transits only
        # preferred_method == "Eclipse"  → eclipses only
        # preferred_method == "Either"   → both (or anything unrecognised)
        preferred = (
            targets["preferred_method"]
            if "preferred_method" in targets.columns
            else pd.Series([METHOD_TRANSIT] * n)
        )
        # Eclipse-only targets must NOT produce a transit candidate.
        has_tr_method = ~preferred.isin([METHOD_ECLIPSE]).to_numpy()
        self._has_transit = has_tr_method & np.isfinite(tr_dur_s) & (tr_dur_s > 0)

        has_ec_method = preferred.isin([METHOD_ECLIPSE, METHOD_EITHER]).to_numpy()
        self._has_eclipse = has_ec_method & np.isfinite(ec_dur_s) & (ec_dur_s > 0)

        # Static science value per event type.
        sw = targets["science_weight"].to_numpy(dtype=float)
        rows = list(targets.itertuples(index=False))
        self._bsv_tr = np.array([
            _base_science_value(targets.iloc[i], sw[i], float(tr_dur_s[i]))
            if self._has_transit[i] else 0.0
            for i in range(n)
        ])
        self._bsv_ec = np.array([
            _base_science_value(targets.iloc[i], sw[i], float(ec_dur_s[i]))
            if self._has_eclipse[i] else 0.0
            for i in range(n)
        ])

        # Tier goals (max tier achievable per target).
        self._tier_goals = targets["max_tier"].to_numpy(dtype=int)

        # Per-step candidate cache: event_id → row dict.
        self._event_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # EventBackend interface
    # ------------------------------------------------------------------

    def candidates(
        self, t_now: float, k: int, unstarted_only: bool = False,
    ) -> pd.DataFrame:
        periods = self._periods

        cf = self._cost_factor   # = 2.5; block extends 1.25 × T14 on each side
        half_tr_block = self._half_tr * cf                # = 1.25 × T14
        half_ec_block = self._half_ec * cf

        # ---- Transits ----
        tr_phase = (t_now - self._epochs) % periods
        if unstarted_only:
            # Next mid whose observation block has not opened yet.
            dt_mid = (periods - tr_phase) % periods
            nxt = t_now + dt_mid
            tr_center = np.where(dt_mid <= half_tr_block, nxt + periods, nxt)
        else:
            in_tr_block = tr_phase < half_tr_block
            tr_center = np.where(
                in_tr_block,
                t_now - tr_phase,
                t_now + (periods - tr_phase),
            )
        tr_ok = self._has_transit & ((tr_center + half_tr_block) > t_now)

        # ---- Eclipses (centred at epoch + period/2 for circular orbits) ----
        ec_epoch = self._epochs + periods / 2.0
        ec_phase = (t_now - ec_epoch) % periods
        if unstarted_only:
            dt_mid = (periods - ec_phase) % periods
            nxt = t_now + dt_mid
            ec_center = np.where(dt_mid <= half_ec_block, nxt + periods, nxt)
        else:
            in_ec_block = ec_phase < half_ec_block
            ec_center = np.where(
                in_ec_block,
                t_now - ec_phase,
                t_now + (periods - ec_phase),
            )
        ec_ok = self._has_eclipse & ((ec_center + half_ec_block) > t_now)

        tr_idx = np.where(tr_ok)[0]   # indices of valid transit targets
        ec_idx = np.where(ec_ok)[0]   # indices of valid eclipse targets
        n_tr, n_ec = len(tr_idx), len(ec_idx)
        n_total = n_tr + n_ec

        if n_total == 0:
            self._event_cache = {}
            return _EMPTY_EVENTS.copy()

        # Combine window_mid times and partial-sort to find K nearest.
        all_mids = np.empty(n_total)
        all_mids[:n_tr] = tr_center[tr_idx]
        all_mids[n_tr:] = ec_center[ec_idx]

        if n_total <= k:
            order = np.argsort(all_mids)
        else:
            # O(N) partial sort, then sort only the K selected.
            part = np.argpartition(all_mids, k)[:k]
            order = part[np.argsort(all_mids[part])]

        # Build result rows and populate per-step cache.
        self._event_cache = {}
        records: list[dict] = []

        for rank in order:
            if rank < n_tr:
                i        = int(tr_idx[rank])
                eid      = i * 2
                half_raw = self._half_tr[i]           # T14/2 (raw transit half)
                mid      = tr_center[i]
                dur_days = self._tr_dur_days[i]
                rec  = {
                    "event_id":              eid,
                    "target_id":             self._target_ids[i],
                    "event_type":            "transit",
                    "window_start":          mid - half_raw,
                    "window_mid":            mid,
                    "window_end":            mid + half_raw,  # raw transit end
                    "duration":              self._tr_dur_s[i],
                    "duration_days":         dur_days,
                    "block_duration_days":   self._cost_factor * dur_days,
                    "tier_goal":             int(self._tier_goals[i]),
                    "base_science_value":    float(self._bsv_tr[i]),
                    "visibility_valid":      True,
                    "ephemeris_uncertainty": 0.0,
                    "event_index":           -1,
                }
            else:
                j        = int(rank - n_tr)
                i        = int(ec_idx[j])
                eid      = i * 2 + 1
                half_raw = self._half_ec[i]
                mid      = ec_center[i]
                dur_days = self._ec_dur_days[i]
                rec  = {
                    "event_id":              eid,
                    "target_id":             self._target_ids[i],
                    "event_type":            "eclipse",
                    "window_start":          mid - half_raw,
                    "window_mid":            mid,
                    "window_end":            mid + half_raw,  # raw eclipse end
                    "duration":              self._ec_dur_s[i],
                    "duration_days":         dur_days,
                    "block_duration_days":   self._cost_factor * dur_days,
                    "tier_goal":             int(self._tier_goals[i]),
                    "base_science_value":    float(self._bsv_ec[i]),
                    "visibility_valid":      True,
                    "ephemeris_uncertainty": 0.0,
                    "event_index":           -1,
                }

            self._event_cache[eid] = rec
            records.append(rec)

        return pd.DataFrame(records)

    def get_event(self, event_id: int) -> pd.Series:
        if event_id not in self._event_cache:
            raise KeyError(
                f"event_id {event_id!r} is not in the DynamicBackend step cache. "
                "candidates() must be called before get_event() each step."
            )
        return pd.Series(self._event_cache[event_id])

    def register_event(self, event_dict: dict) -> int:
        """Register an event from ``events_for_target`` in the step cache.

        Assigns the canonical event_id for this target/event_type combination
        (``target_idx * 2`` for transit, ``* 2 + 1`` for eclipse), overwrites
        any existing cache entry for that ID, and returns the id.

        This allows ``full_set`` candidate selection to cache events that were
        not included in the top-K ``candidates()`` pool (e.g. long-period targets
        whose next occurrence is far in the future) so that
        ``execute_observation`` can look them up via ``get_event``.
        """
        target_id = str(event_dict.get("target_id", ""))
        idx = self._tid_to_idx.get(target_id)
        if idx is None:
            return -1
        etype = str(event_dict.get("event_type", "transit"))
        eid = idx * 2 if etype == "transit" else idx * 2 + 1
        # Store a complete record (with event_id set correctly)
        cached = {**event_dict, "event_id": eid}
        # Fill in any fields that execute_observation might need
        cached.setdefault("tier_goal",              self._tier_goals[idx])
        cached.setdefault("base_science_value",     1.0)
        cached.setdefault("visibility_valid",        True)
        cached.setdefault("ephemeris_uncertainty",   0.0)
        cached.setdefault("event_index",             -1)
        cached.setdefault("duration", cached.get("duration_days", 0.0) * 86400.0)
        self._event_cache[eid] = cached
        return eid

    def events_for_target(
        self,
        target_id: str,
        t_now: float,
        n: int = 3,
    ) -> list[dict]:
        """Return the next *n* events for *target_id* starting from *t_now*.

        Respects ``preferred_method``: transit-only targets return only transit
        events; eclipse-only targets return only eclipse events; either-type
        targets interleave the nearest events from both.

        The first event is always the current occurrence if the block is still
        open (consistent with ``candidates()``), or the next future occurrence.
        """
        idx = self._tid_to_idx.get(target_id)
        if idx is None:
            return []

        period    = float(self._periods[idx])
        cf        = self._cost_factor

        events_out: list[dict] = []

        def _next_mids_from(epoch: float, half_event: float, n_events: int) -> list[float]:
            """Return next n_events mid-times at or after t_now."""
            half_block = half_event * cf
            phase      = (t_now - epoch) % period
            if phase < half_block:                  # still inside current block
                first_mid = t_now - phase
            else:
                first_mid = t_now + (period - phase)
            return [first_mid + k * period for k in range(n_events)]

        want_transit = self._has_transit[idx]
        want_eclipse = self._has_eclipse[idx]

        if want_transit and not want_eclipse:
            mids = _next_mids_from(
                self._epochs[idx], self._half_tr[idx], n
            )
            dur_s    = float(self._tr_dur_s[idx])
            dur_days = float(self._tr_dur_days[idx])
            for mid in mids:
                events_out.append({
                    "window_mid":          mid,
                    "window_start":        mid - self._half_tr[idx],
                    "window_end":          mid + self._half_tr[idx],
                    "duration_days":       dur_days,
                    "block_duration_days": cf * dur_days,
                    "event_type":          "transit",
                })
        elif want_eclipse and not want_transit:
            ec_epoch = self._epochs[idx] + period / 2.0
            mids = _next_mids_from(ec_epoch, self._half_ec[idx], n)
            dur_s    = float(self._ec_dur_s[idx])
            dur_days = float(self._ec_dur_days[idx])
            for mid in mids:
                events_out.append({
                    "window_mid":          mid,
                    "window_start":        mid - self._half_ec[idx],
                    "window_end":          mid + self._half_ec[idx],
                    "duration_days":       dur_days,
                    "block_duration_days": cf * dur_days,
                    "event_type":          "eclipse",
                })
        elif want_transit and want_eclipse:
            # Interleave both types; return the n nearest.
            tr_mids = _next_mids_from(
                self._epochs[idx], self._half_tr[idx], n
            )
            ec_epoch = self._epochs[idx] + period / 2.0
            ec_mids = _next_mids_from(ec_epoch, self._half_ec[idx], n)

            tr_dur_days = float(self._tr_dur_days[idx])
            ec_dur_days = float(self._ec_dur_days[idx])

            combined = (
                [(m, "transit", self._half_tr[idx], tr_dur_days) for m in tr_mids]
                + [(m, "eclipse", self._half_ec[idx], ec_dur_days) for m in ec_mids]
            )
            combined.sort(key=lambda x: x[0])
            for mid, etype, half_raw, dur_d in combined[:n]:
                events_out.append({
                    "window_mid":          mid,
                    "window_start":        mid - half_raw,
                    "window_end":          mid + half_raw,
                    "duration_days":       dur_d,
                    "block_duration_days": cf * dur_d,
                    "event_type":          etype,
                })
        # If neither (target has no valid events), return empty list
        return events_out[:n]

    def export_events(self, t_start: float, t_end: float) -> pd.DataFrame:
        """Materialise every transit/eclipse whose mid-time is in ``[t_start, t_end]``.

        Uses the same circular-orbit model as :meth:`candidates`.  Intended
        for analysis and plotting — the RL loop does not need this table.
        """
        if t_end <= t_start:
            return _EMPTY_EVENTS.copy()

        records: list[dict] = []
        event_id = 0
        cf = self._cost_factor

        def _indices(epoch: float, period: float) -> np.ndarray:
            n0 = int(np.ceil((t_start - epoch) / period - 1e-12))
            n1 = int(np.floor((t_end - epoch) / period + 1e-12))
            if n1 < n0:
                return np.empty(0, dtype=int)
            return np.arange(n0, n1 + 1)

        for i in range(len(self._target_ids)):
            period = float(self._periods[i])
            if not np.isfinite(period) or period <= 0:
                continue
            tid = self._target_ids[i]
            tier = int(self._tier_goals[i])

            if self._has_transit[i]:
                half = float(self._half_tr[i])
                dur_days = float(self._tr_dur_days[i])
                for n in _indices(float(self._epochs[i]), period):
                    mid = float(self._epochs[i]) + int(n) * period
                    records.append({
                        "event_id": event_id,
                        "target_id": tid,
                        "event_type": "transit",
                        "window_start": mid - half,
                        "window_mid": mid,
                        "window_end": mid + half,
                        "duration": float(self._tr_dur_s[i]),
                        "duration_days": dur_days,
                        "block_duration_days": cf * dur_days,
                        "tier_goal": tier,
                        "base_science_value": float(self._bsv_tr[i]),
                        "visibility_valid": True,
                        "ephemeris_uncertainty": 0.0,
                        "event_index": int(n),
                    })
                    event_id += 1

            if self._has_eclipse[i]:
                half = float(self._half_ec[i])
                dur_days = float(self._ec_dur_days[i])
                ec_epoch = float(self._epochs[i]) + period / 2.0
                for n in _indices(ec_epoch, period):
                    mid = ec_epoch + int(n) * period
                    records.append({
                        "event_id": event_id,
                        "target_id": tid,
                        "event_type": "eclipse",
                        "window_start": mid - half,
                        "window_mid": mid,
                        "window_end": mid + half,
                        "duration": float(self._ec_dur_s[i]),
                        "duration_days": dur_days,
                        "block_duration_days": cf * dur_days,
                        "tier_goal": tier,
                        "base_science_value": float(self._bsv_ec[i]),
                        "visibility_valid": True,
                        "ephemeris_uncertainty": 0.0,
                        "event_index": int(n),
                    })
                    event_id += 1

        if not records:
            return _EMPTY_EVENTS.copy()
        return (
            pd.DataFrame.from_records(records)
            .sort_values("window_mid")
            .reset_index(drop=True)
        )

    def reset(self) -> None:
        self._event_cache.clear()

    def add_targets(self, new_targets: pd.DataFrame) -> None:
        """Append newly confirmed targets to the orbital arrays.

        Rebuilds the backend arrays so subsequent ``candidates()`` /
        ``events_for_target()`` calls include the new planets.  Existing
        target indices are preserved (appended only).
        """
        if new_targets is None or len(new_targets) == 0:
            return

        from aRieL.simulator.event_generator import _base_science_value
        from aRieL.data.schemas import METHOD_ECLIPSE, METHOD_EITHER, METHOD_TRANSIT

        n0 = len(self._target_ids)
        n_new = len(new_targets)
        new_ids = new_targets["target_id"].astype(str).to_numpy()

        self._target_ids = np.concatenate([self._target_ids, new_ids])
        for j, tid in enumerate(new_ids):
            self._tid_to_idx[str(tid)] = n0 + j

        self._epochs  = np.concatenate([
            self._epochs, new_targets["epoch"].to_numpy(dtype=float)
        ])
        self._periods = np.concatenate([
            self._periods, new_targets["period"].to_numpy(dtype=float)
        ])

        tr_dur_s = new_targets["transit_duration"].to_numpy(dtype=float)
        self._tr_dur_s    = np.concatenate([self._tr_dur_s, tr_dur_s])
        self._tr_dur_days = np.concatenate([self._tr_dur_days, tr_dur_s / 86400.0])
        self._half_tr     = np.concatenate([self._half_tr, tr_dur_s / 86400.0 / 2.0])

        ec_dur_s = new_targets["eclipse_duration"].to_numpy(dtype=float)
        missing_ec = ~np.isfinite(ec_dur_s) | (ec_dur_s <= 0)
        ec_dur_s = np.where(missing_ec, tr_dur_s, ec_dur_s)
        self._ec_dur_s    = np.concatenate([self._ec_dur_s, ec_dur_s])
        self._ec_dur_days = np.concatenate([self._ec_dur_days, ec_dur_s / 86400.0])
        self._half_ec     = np.concatenate([self._half_ec, ec_dur_s / 86400.0 / 2.0])

        preferred = (
            new_targets["preferred_method"]
            if "preferred_method" in new_targets.columns
            else pd.Series([METHOD_TRANSIT] * n_new)
        )
        has_tr_method = ~preferred.isin([METHOD_ECLIPSE]).to_numpy()
        has_ec_method = preferred.isin([METHOD_ECLIPSE, METHOD_EITHER]).to_numpy()
        new_has_tr = has_tr_method & np.isfinite(tr_dur_s) & (tr_dur_s > 0)
        new_has_ec = has_ec_method & np.isfinite(ec_dur_s) & (ec_dur_s > 0)
        self._has_transit = np.concatenate([self._has_transit, new_has_tr])
        self._has_eclipse = np.concatenate([self._has_eclipse, new_has_ec])

        sw = new_targets["science_weight"].to_numpy(dtype=float)
        bsv_tr = np.zeros(n_new, dtype=float)
        bsv_ec = np.zeros(n_new, dtype=float)
        for j in range(n_new):
            row = new_targets.iloc[j]
            if new_has_tr[j]:
                bsv_tr[j] = _base_science_value(row, sw[j], float(tr_dur_s[j]))
            if new_has_ec[j]:
                bsv_ec[j] = _base_science_value(row, sw[j], float(ec_dur_s[j]))
        self._bsv_tr = np.concatenate([self._bsv_tr, bsv_tr])
        self._bsv_ec = np.concatenate([self._bsv_ec, bsv_ec])

        self._tier_goals = np.concatenate([
            self._tier_goals, new_targets["max_tier"].to_numpy(dtype=int)
        ])
        # Drop step cache — indices for new targets are fresh; old IDs unchanged.
        self._event_cache.clear()
