"""RDE Pipeline — saare modules ko ek flow me jodta hai.

  Raw files -> Cleaner -> Validator -> Quality -> Repository -> Language
            -> Encoder -> FIM Builder -> Curriculum -> Chunk Builder -> Dedup
            -> Shards (RDS) -> Manifest Lock -> Validation Report

Supported input layouts (both work):
  1. Multi-repo : `root/<repo_name>/<files...>`  (har top-level folder = ek repo)
  2. Flat       : `root/<files...>`              (root itself = ek repo)

If both a flat file and subdirectories exist under root, the multi-repo layout is
used AND the loose files under root are processed as a repo named after `root`.

Output: `out_dir/shard_00000.rds ...` + `manifest.json` + `report.{json,html}`.
"""

from __future__ import annotations

import os


class EmptyDatasetError(RuntimeError):
    """Raised when discovery/processing yields zero usable files or chunks.

    We never silently write an empty dataset — that hides real problems (wrong
    path, wrong layout, over-aggressive filters).
    """

from .chunker import ChunkBuilder
from .cleaner import Cleaner
from .config import RDEConfig
from .curriculum import CurriculumBuilder
from .dedup import ChunkDeduper
from .encoder import Encoder
from .fim import FIMBuilder
from .language import detect_language
from .lock import build_lock
from .quality import QualityAnalyzer
from .record import FileRecord
from .report import build_report, write_report
from .repository import RepositoryAnalyzer
from .sharding import ShardManager
from .stats import StatsCollector
from .validator import Validator


class RDEPipeline:
    def __init__(self, tokenizer, config: RDEConfig | None = None):
        self.tok = tokenizer
        self.cfg = config or RDEConfig()
        self.cleaner = Cleaner(self.cfg)
        self.validator = Validator(self.cfg)
        self.quality = QualityAnalyzer()
        self.repo_analyzer = RepositoryAnalyzer()
        self.encoder = Encoder(tokenizer, self.cfg)
        self.fim = FIMBuilder(self.cfg)
        self.curriculum = CurriculumBuilder()
        self.deduper = ChunkDeduper()
        self.stats = StatsCollector(self.cfg.seq_len, self.cfg.token_itemsize)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _discover_repos(self, root: str) -> list[tuple[str, str, set]]:
        """Return [(repo_name, repo_root, prune_dirs), ...], supporting both layouts.

        - Each subdirectory of `root` is a repo (multi-repo layout).
        - If `root` also holds loose files directly, `root` itself is added as a
          repo named after its basename (flat layout). Its walk prunes the
          top-level subdirs that are already their own repos, so no file is
          processed twice.
        Vendor dirs are skipped at this top level so an all-vendor root is treated
        as flat rather than silently empty.
        """
        if not os.path.isdir(root):
            raise EmptyDatasetError(
                f"input path is not a directory: {root!r}\n"
                f"Expected either:\n"
                f"  {root}/<repo>/<files...>   (multi-repo layout), or\n"
                f"  {root}/<files...>          (flat layout).")

        entries = sorted(os.listdir(root))
        subdirs = [d for d in entries
                   if os.path.isdir(os.path.join(root, d))
                   and d not in self.cfg.vendor_dirs]
        has_loose_files = any(
            os.path.isfile(os.path.join(root, e)) for e in entries)

        repos: list[tuple[str, str, set]] = [
            (d, os.path.join(root, d), set()) for d in subdirs]

        # Flat layout: loose files directly under root (or a root with nothing
        # but vendor dirs) -> treat root itself as a repo. Prune the subdirs that
        # are already repos so their files aren't discovered twice.
        if has_loose_files or not repos:
            repos.append((os.path.basename(os.path.abspath(root)) or "root",
                          root, set(subdirs)))
        return repos

    def _walk_repo(self, repo_root: str, prune_dirs: set):
        """Yield (full_path, rel_path) for files in a repo.

        `prune_dirs` (top-level directory names) are not descended into — used so
        the flat root-repo skips subdirs that are already separate repos. Vendor
        dirs are additionally dropped here (Cleaner also drops them by path).
        """
        for dirpath, dirnames, files in os.walk(repo_root):
            if dirpath == repo_root:
                dirnames[:] = [d for d in dirnames
                               if d not in prune_dirs
                               and d not in self.cfg.vendor_dirs]
            for fn in sorted(files):
                full = os.path.join(dirpath, fn)
                yield full, os.path.relpath(full, repo_root)

    def run(self, root: str, out_dir: str, verbose: bool = True,
            now_iso: str | None = None, debug: bool = False) -> dict:
        records = []

        repos = self._discover_repos(root)
        if debug:
            print(f"[discover] root={root!r} -> repos="
                  f"{[name for name, _, _ in repos]}")

        for repo, repo_root, prune_dirs in repos:
            info = self.repo_analyzer.analyze(repo, repo_root)
            repo_meta = info.to_meta()

            for full, rel in self._walk_repo(repo_root, prune_dirs):
                self.stats.total_files_seen += 1
                if debug:
                    print(f"[discover] file: {repo}/{rel}")
                try:
                    with open(full, "rb") as fh:
                        raw = fh.read()
                except OSError as exc:
                    if debug:
                        print(f"[read]     SKIP {rel}: {exc}")
                    continue

                # 1) CLEAN
                rec = self.cleaner.clean_bytes(rel, repo, raw)
                if rec is None:
                    if debug:
                        print(f"[cleaner]  DROP {rel}")
                    continue
                # 5) LANGUAGE
                rec.language = detect_language(rec.path, rec.text)
                if debug:
                    print(f"[language] {rel} -> {rec.language}")
                # 2) VALIDATE
                if not self.validator.validate(rec):
                    self.stats.file_drops[rec.drop_reason] += 1
                    if debug:
                        print(f"[validate] DROP {rel}: {rec.drop_reason}")
                    continue
                # 3) QUALITY
                self.quality.score(rec, repo_meta)
                if rec.quality < self.cfg.min_quality:
                    self.stats.file_drops["low_quality"] += 1
                    if debug:
                        print(f"[quality]  DROP {rel}: score {rec.quality}"
                              f" < min {self.cfg.min_quality}")
                    continue
                # 7) ENCODE
                self.encoder.encode_record(rec)
                if debug:
                    print(f"[encoder]  {rel}: quality={rec.quality} "
                          f"tokens={len(rec.token_ids)}")
                rec.meta.update({"license": repo_meta["license"],
                                 "stars": repo_meta["stars"], "kind": "code"})
                records.append(rec)

                # 4/FIM) is file ke functions se FIM examples banao
                fim = self._fim_records(rec)
                if debug and fim:
                    print(f"[fim]      {rel}: +{len(fim)} FIM example(s)")
                records.extend(fim)

        # No usable files? Fail loudly — never write an empty dataset.
        if not records:
            raise EmptyDatasetError(self._empty_message(root))

        # 9) CURRICULUM (easy -> hard, multi-signal)
        records = self.curriculum.order(records)
        for rec in records:
            self.stats.add_file(rec)
        if debug:
            print(f"[curriculum] {len(records)} record(s) ordered easy->hard")

        # 8) CHUNK + 11) DEDUP + 15) SHARD + packing stats
        chunker = ChunkBuilder(self.cfg)
        chunker.set_pad_id(getattr(self.tok, "pad_id", 0))
        shards = ShardManager(out_dir, self.cfg, self.tok)

        n_written = 0
        for chunk in chunker.build(records):
            if self.cfg.dedup_chunks and self.deduper.is_duplicate(chunk.token_ids):
                continue
            shards.add_chunk(chunk.token_ids, chunk.meta)
            self.stats.add_chunk(chunk)
            n_written += 1
        if debug:
            print(f"[chunker]  built chunks; {n_written} written after dedup "
                  f"({self.deduper.n_dropped} duplicate(s) dropped)")

        # Guard: files existed but produced no chunks (shouldn't happen, but never
        # write an empty dataset silently).
        if n_written == 0:
            raise EmptyDatasetError(
                f"processed {len(records)} record(s) but produced 0 chunks — "
                f"nothing to write. Check seq_len ({self.cfg.seq_len}) and dedup "
                f"settings.")

        self.stats.chunk_dupe_seen = self.deduper.n_seen
        self.stats.chunk_dupe_dropped = self.deduper.n_dropped

        # Manifest Lock (reproducibility) — finalize se pehle banao
        lock = build_lock(self.cfg, self.tok,
                          dataset_version=self.cfg.dataset_version,
                          model_version=self.cfg.model_version,
                          source_root=root, now_iso=now_iso)
        manifest = shards.finalize(self.stats.summary(), lock=lock)

        # Validation Report (JSON + HTML)
        report = build_report(manifest, self.cleaner.stats)
        write_report(report, out_dir)

        if verbose:
            self._report(manifest)
        return manifest

    # ------------------------------------------------------------------ #
    def _empty_message(self, root: str) -> str:
        """Build a descriptive error when zero usable files are found."""
        seen = self.stats.total_files_seen
        drops = dict(self.cleaner.stats)
        vdrops = dict(self.stats.file_drops)
        lines = [
            f"No usable files were found under {root!r}; refusing to write an "
            f"empty dataset.",
            "",
            f"Files discovered: {seen}",
        ]
        if seen == 0:
            lines += [
                "",
                "Nothing was discovered. Check the input path and layout. RDE "
                "accepts either:",
                f"  {root}/<repo>/<files...>   (multi-repo layout), or",
                f"  {root}/<files...>          (flat layout).",
            ]
        else:
            lines += [
                f"Cleaner drops : {drops}",
                f"Validator drops: {vdrops}",
                "",
                "All discovered files were filtered out. Common causes: files are "
                "vendored/generated/binary/duplicate, fail syntax/encoding checks, "
                "or have unsupported extensions/languages.",
            ]
        return "\n".join(lines)

    def _fim_records(self, rec: FileRecord) -> list[FileRecord]:
        """Code file ke functions se FIM FileRecords banao (source difficulty inherit)."""
        out = []
        for i, fim_text in enumerate(self.fim.build_for(rec)):
            fr = FileRecord(path=f"{rec.path}#fim{i}", repo=rec.repo,
                            text=fim_text, language="python")
            self.encoder.encode_record(fr)
            fr.difficulty = rec.difficulty
            fr.quality = rec.quality
            fr.meta["kind"] = "fim"
            out.append(fr)
        return out

    def _report(self, manifest):
        s = manifest["stats"]
        pk = s["packing"]
        print("=" * 56)
        print("RDE pipeline complete")
        print("=" * 56)
        print(f"  files seen     : {s['total_files_seen']}")
        print(f"  files kept     : {s['files_kept']}  (fim docs: {s['fim_docs']})")
        print(f"  clean drops    : {self.cleaner.stats}")
        print(f"  validate drops : {s['file_drop_reasons']}")
        print(f"  languages      : {s['languages']}")
        print(f"  difficulty     : {s['difficulty_split']}")
        print(f"  avg quality    : {s['avg_quality']}")
        print(f"  total chunks   : {s['total_chunks']}  (tokens: {s['total_tokens']})")
        print(f"  packing eff.   : {pk['packing_efficiency_pct']}%  "
              f"(padding {pk['padding_pct']}%, avg ctx {pk['avg_context_usage']}/{pk['seq_len']})")
        print(f"  compression    : {s['compression_ratio']}x")
        print(f"  chunk dupe %   : {s['chunk_duplicate_pct']}  "
              f"(backend={self.deduper.backend})")
        print(f"  shards         : {manifest['n_shards']}  (dtype={manifest['dtype']})")
        print(f"  report         : {os.path.join('', 'report.html')} written")
