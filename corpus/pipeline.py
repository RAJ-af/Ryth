"""CorpusPipeline — orchestrate the full corpus build.

Stages:
  ingest -> clean -> language annotate -> license filter -> exact dedup
        -> near dedup -> repo dedup -> quality score/threshold -> split
        -> (optional) language-ratio balance

Har stage records ke `drop_reason` set karta hai; end me kept + dropped separate
ho jaate hain aur drop reasons ka count report ke liye milta hai. Timestamps caller
se aate hain (`created_at`) — package deterministic rehta hai.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cleaners import Cleaner
from .config import CorpusConfig
from .dedup import dedupe_files, dedupe_near, dedupe_repos
from .download import DownloadError, resolve_downloader
from .filters import LicenseFilter, annotate_language, balance_language_ratios
from .licenses import detect_repo_license
from .licenses.spdx import UNKNOWN
from .metadata import FileRecord, RepoRecord
from .quality import score_repo
from .split import split_records, verify_no_leakage


@dataclass
class CorpusResult:
    records: list = field(default_factory=list)     # kept FileRecords
    repos: list = field(default_factory=list)       # kept RepoRecords
    drops: dict = field(default_factory=dict)       # drop_reason -> count
    all_records: list = field(default_factory=list)  # kept + dropped

    @property
    def n_files(self) -> int:
        return len(self.records)


class CorpusPipeline:
    def __init__(self, config: CorpusConfig | None = None):
        self.cfg = config or CorpusConfig()
        self.cleaner = Cleaner(self.cfg)
        self.license_filter = LicenseFilter(self.cfg)

    # ---- stage 1: ingest + clean ----
    def ingest(self, sources, stage_dir: str, created_at: str = "") -> list:
        records: list = []
        for source in sources:
            dl = resolve_downloader(source.kind)
            if not dl.available():
                continue
            try:
                staged = dl.fetch(source, stage_dir)
            except DownloadError:
                continue
            lic = detect_repo_license(staged.read_text_files())
            if lic == UNKNOWN:
                lic = source.license_hint
            for path, data in staged.iter_files():
                res = self.cleaner.inspect(path, data)
                rec = FileRecord.from_bytes(
                    staged.repo, path, data, source=staged.source,
                    license=lic, created_at=created_at)
                if not res.keep:
                    rec.drop_reason = res.reason
                    rec.content = None
                else:
                    rec.content = res.text
                records.append(rec)
        return records

    # ---- stages 2..N: process in-memory records ----
    def process(self, records: list) -> CorpusResult:
        cfg = self.cfg

        def kept():
            return [r for r in records if not r.drop_reason]

        # language + license
        for r in kept():
            annotate_language(r)
        for r in kept():
            ok, reason = self.license_filter.check(r)
            if not ok:
                r.drop_reason = reason

        # exact file dedup
        if cfg.dedup_files:
            _, dropped = dedupe_files(kept())
            # dedupe_files already set drop_reason on dropped
            _ = dropped

        # near-duplicate dedup
        if cfg.near_dedup:
            dedupe_near(kept(), cfg.minhash_perms, cfg.minhash_bands,
                        cfg.near_dup_threshold)

        # repo-level exact dedup
        if cfg.dedup_repos:
            repo_hashes: dict = {}
            for r in kept():
                repo_hashes.setdefault(r.repository, []).append(r.hash)
            keep_map = dedupe_repos(repo_hashes)
            for r in kept():
                if not keep_map.get(r.repository, True):
                    r.drop_reason = "duplicate_repo"

        # quality scoring + threshold (repo level)
        repos: list = []
        by_repo: dict = {}
        for r in kept():
            by_repo.setdefault(r.repository, []).append(r)
        for repo, recs in by_repo.items():
            score, signals = score_repo(recs)
            if score < cfg.min_quality:
                for r in recs:
                    r.drop_reason = "low_quality"
                continue
            langs: dict = {}
            for r in recs:
                langs[r.language] = langs.get(r.language, 0) + 1
                r.quality_score = score
                r.file_quality = score
            repos.append(RepoRecord(
                repository=repo, source=recs[0].source, license=recs[0].license,
                quality_score=score, n_files=len(recs),
                n_bytes=sum(r.size for r in recs), languages=langs,
                split=recs[0].split, signals=signals,
                created_at=recs[0].created_at))

        # split (repo-level, deterministic)
        split_records(kept(), cfg.split_ratios, cfg.seed)
        for rp in repos:
            recs = by_repo[rp.repository]
            rp.split = recs[0].split

        # optional language-ratio balancing
        if cfg.enforce_language_ratios:
            keep_set = {id(r) for r in
                        balance_language_ratios(kept(), cfg.language_ratios)}
            for r in kept():
                if id(r) not in keep_set:
                    r.drop_reason = "language_ratio"

        final = [r for r in records if not r.drop_reason]
        drops: dict = {}
        for r in records:
            if r.drop_reason:
                drops[r.drop_reason] = drops.get(r.drop_reason, 0) + 1
        # keep only repos that still have kept files
        alive = {r.repository for r in final}
        repos = [rp for rp in repos if rp.repository in alive]
        return CorpusResult(records=final, repos=repos, drops=drops,
                            all_records=records)

    # ---- convenience: full build ----
    def build(self, sources, stage_dir: str, created_at: str = "") -> CorpusResult:
        raw = self.ingest(sources, stage_dir, created_at)
        result = self.process(raw)
        assert verify_no_leakage(result.records), "repository split leakage detected"
        return result
