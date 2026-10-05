from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from scipy.io import arff


@dataclass(frozen=True)
class GEOSpec:
    accession: str
    public_name: str
    expected_samples: int
    expected_features: int
    expected_classes: int
    grouped: bool
    source_note: str


GEO_SPECS: dict[str, GEOSpec] = {
    "GSE3141": GEOSpec(
        "GSE3141", "Lung_cancer", 111, 54675, 2, False,
        "Cell type A/S parsed from GEO sample characteristics.",
    ),
    "GSE4290": GEOSpec(
        "GSE4290", "Brain_Tumor", 180, 54613, 4, False,
        "Four pathology classes. Four blank diagnostic fields are assigned to glioblastoma to match the official 81-GBM series total.",
    ),
    "GSE3726": GEOSpec(
        "GSE3726", "Breast_cancer_2", 104, 22283, 2, True,
        "Breast/colon labels and matched RNAlater/frozen pair IDs parsed from GEO titles.",
    ),
    "GSE15471": GEOSpec(
        "GSE15471", "Pancreatic_Cancer", 78, 54675, 2, True,
        "Normal/tumor labels, patient IDs, and technical replicate groups parsed from GEO titles.",
    ),
    "GSE10072": GEOSpec(
        "GSE10072", "Lung_Adenocarcinoma", 107, 22283, 2, True,
        "Tumor/non-tumor labels and GT subject IDs parsed from GEO titles.",
    ),
}


@dataclass(frozen=True)
class LegacySpec:
    public_name: str
    filename: str
    expected_samples: int
    expected_features: int
    expected_classes: int
    url: str
    sha256: str
    license: str


_GITHUB_COMMIT = "5681201bb4e2ba7024fe15d4c71046017219b658"
_GITHUB_RAW = f"https://raw.githubusercontent.com/kivancguckiran/microarray-data/{_GITHUB_COMMIT}/csv"
_MENDELEY = "https://data.mendeley.com/public-files/datasets/fhx5zgx2zj/files"
LEGACY_SPECS: dict[str, LegacySpec] = {
    "Breast_cancer_1": LegacySpec(
        "Breast_cancer_1", "Breast.arff", 97, 24481, 2,
        f"{_MENDELEY}/c2c65b79-9be4-4728-9c5d-bf913eb4a1bb/file_downloaded",
        "4f0b0ecdf0e1ffc7a1a16fc0b9eabc699aa18ea686b5d7d4ae417428b8a39d1a",
        "CC BY-NC 3.0 (Mendeley dataset DOI 10.17632/fhx5zgx2zj.1)",
    ),
    "SRBCT": LegacySpec(
        "SRBCT", "SRBCT.arff", 83, 2308, 4,
        f"{_MENDELEY}/309860df-6224-4f87-9cac-59c580c283db/file_downloaded",
        "80b1872f2fe61a6a743687b096af7b7bce2aba45ed4a0c9856152d8241de7a9a",
        "CC BY-NC 3.0 (Mendeley dataset DOI 10.17632/fhx5zgx2zj.1)",
    ),
    "11_Tumor": LegacySpec(
        "11_Tumor", "11_Tumors.xlsx", 174, 12533, 11,
        f"{_MENDELEY}/d2729d4b-e97c-457e-9310-d3cbc08fc317/file_downloaded",
        "9088cb447edde7b03068c581d38a5669fc41cb2f7a4683e2b3682bfb087fd913",
        "CC BY-NC 3.0 (Mendeley dataset DOI 10.17632/fhx5zgx2zj.1)",
    ),
    "Colon": LegacySpec(
        "Colon", "alon.tar.gz", 62, 2000, 2,
        f"{_GITHUB_RAW}/alon.tar.gz",
        "caa28ffa83219916d0e3ab9c2e459c0a43e407f5ec55ab49301e69493cef5628",
        "Upstream repository does not declare a data license; download only",
    ),
    "Leukemia": LegacySpec(
        "Leukemia", "golub.tar.gz", 72, 7129, 2,
        f"{_GITHUB_RAW}/golub.tar.gz",
        "4f075c5f16c0c85adb724b75aca762b8285836fe875d2860059fe3096b7c3940",
        "Upstream repository does not declare a data license; download only",
    ),
}


def _bucket(accession: str) -> str:
    return re.sub(r"\d{3}$", "nnn", accession)


def geo_urls(accession: str) -> list[str]:
    name = f"{accession}_series_matrix.txt.gz"
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{_bucket(accession)}/{accession}"
    return [f"{base}/matrix/{name}", f"{base}/suppl/{name}"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_series_matrix(
    accession: str,
    raw_dir: str | Path,
    *,
    retries: int = 3,
) -> tuple[Path, str]:
    accession = accession.upper()
    if accession not in GEO_SPECS:
        raise ValueError(f"unsupported GEO accession: {accession}")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{accession}_series_matrix.txt.gz"
    if target.exists() and target.stat().st_size:
        return target, "existing local file"
    user_agent = "ARFS-E-NSGA-II-reproducibility/0.1 (public research data fetcher)"
    errors: list[str] = []
    for url in geo_urls(accession):
        for attempt in range(retries):
            request = urllib.request.Request(url, headers={"User-Agent": user_agent})
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                temporary.replace(target)
                return target, url
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
                temporary.unlink(missing_ok=True)
                errors.append(f"{url} attempt {attempt + 1}: {error}")
                if attempt + 1 < retries:
                    time.sleep(5 * (attempt + 1))
    raise RuntimeError("GEO download failed:\n" + "\n".join(errors))


def _download_file(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        return target
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ARFS-E-NSGA-II-reproducibility/0.1"},
    )
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _ebi_fallback(
    accession: str, raw_dir: Path
) -> tuple[pd.DataFrame, dict[str, list[list[str]]], list[tuple[Path, str]]]:
    """Load official EMBL-EBI mirrors when the NCBI endpoint is unavailable."""
    if accession == "GSE10072":
        url = (
            "https://www.ebi.ac.uk/biostudies/files/E-GEOD-10072/"
            "E-GEOD-10072-processed-data-1665603668.txt"
        )
        path = _download_file(url, raw_dir / "E-GEOD-10072-processed-data.txt")
        matrix = pd.read_csv(path, sep="\t", index_col=0, skiprows=[1], low_memory=False)
        titles = matrix.columns.astype(str).tolist()
        metadata = {
            "!Sample_title": [titles],
            "!Sample_source_name_ch1": [[""] * len(titles)],
            "!Sample_characteristics_ch1": [[""] * len(titles)],
        }
        return matrix, metadata, [(path, url)]
    if accession == "GSE15471":
        base = "https://www.ebi.ac.uk/biostudies/files/E-GEOD-15471"
        archive_url = (
            "https://ftp.ebi.ac.uk/pub/databases/microarray/data/experiment/GEOD/"
            "E-GEOD-15471/E-GEOD-15471.processed.1.zip"
        )
        sdrf_url = f"{base}/E-GEOD-15471.sdrf.txt"
        archive = _download_file(archive_url, raw_dir / "E-GEOD-15471.processed.1.zip")
        sdrf_path = _download_file(sdrf_url, raw_dir / "E-GEOD-15471.sdrf.txt")
        sdrf = pd.read_csv(sdrf_path, sep="\t", dtype=str)
        file_column = "Derived Array Data File"
        title_column = "Comment [Sample_title]"
        if file_column not in sdrf or title_column not in sdrf:
            raise ValueError("unexpected E-GEOD-15471 SDRF schema")
        mapping = dict(zip(sdrf[file_column], sdrf[title_column]))
        columns: dict[str, pd.Series] = {}
        with zipfile.ZipFile(archive) as bundle:
            members = {Path(name).name: name for name in bundle.namelist()}
            for filename, title in mapping.items():
                if filename not in members:
                    raise ValueError(f"missing processed table in EBI archive: {filename}")
                with bundle.open(members[filename]) as handle:
                    table = pd.read_csv(handle, sep="\t", index_col=0)
                if "VALUE" not in table:
                    raise ValueError(f"missing VALUE column in {filename}")
                columns[str(title)] = table["VALUE"]
        matrix = pd.DataFrame(columns)
        titles = matrix.columns.tolist()
        metadata = {
            "!Sample_title": [titles],
            "!Sample_source_name_ch1": [[""] * len(titles)],
            "!Sample_characteristics_ch1": [[""] * len(titles)],
        }
        return matrix, metadata, [(archive, archive_url), (sdrf_path, sdrf_url)]
    raise ValueError(f"no EBI fallback configured for {accession}")


def read_geo_metadata(path: str | Path) -> dict[str, list[list[str]]]:
    metadata: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("!Sample_"):
                row = next(csv.reader([line], delimiter="\t"))
                metadata.setdefault(row[0], []).append(row[1:])
            if line.startswith("!series_matrix_table_begin"):
                break
    return metadata


def _metadata_row(metadata: dict[str, list[list[str]]], key: str, row: int = 0) -> list[str]:
    try:
        return metadata[key][row]
    except (KeyError, IndexError) as error:
        raise ValueError(f"missing GEO metadata row: {key}[{row}]") from error


def labels_and_groups(
    accession: str, metadata: dict[str, list[list[str]]]
) -> tuple[list[str], list[str] | None]:
    titles = _metadata_row(metadata, "!Sample_title")
    sources = _metadata_row(metadata, "!Sample_source_name_ch1")
    characteristics = _metadata_row(metadata, "!Sample_characteristics_ch1")
    if accession == "GSE3141":
        labels = []
        for value in characteristics:
            match = re.search(r"Cell type:\s*([AS])(?:;|$)", value, re.IGNORECASE)
            if not match:
                raise ValueError(f"cannot parse GSE3141 class: {value}")
            labels.append(match.group(1).upper())
        return labels, None
    if accession == "GSE4290":
        labels = []
        for value in characteristics:
            lowered = value.lower()
            if "non-tumor" in lowered:
                labels.append("non_tumor")
            elif "astrocytoma" in lowered:
                labels.append("astrocytoma")
            elif "oligodendroglioma" in lowered:
                labels.append("oligodendroglioma")
            elif "glioblastoma" in lowered or lowered.rstrip().endswith("diagnostic:"):
                labels.append("glioblastoma")
            else:
                raise ValueError(f"cannot parse GSE4290 class: {value}")
        expected = {
            "non_tumor": 23,
            "astrocytoma": 26,
            "oligodendroglioma": 50,
            "glioblastoma": 81,
        }
        if pd.Series(labels).value_counts().to_dict() != expected:
            raise ValueError("GSE4290 pathology counts do not match the official series design")
        return labels, None
    if accession == "GSE3726":
        labels = [source.strip().lower() for source in sources]
        groups = [re.sub(r"[RT]$", "", title, flags=re.IGNORECASE) for title in titles]
        return labels, groups
    if accession == "GSE15471":
        labels = ["normal" if title.upper().startswith("N") else "tumor" for title in titles]
        groups = [re.sub(r"_rep$", "", title[1:], flags=re.IGNORECASE) for title in titles]
        return labels, groups
    if accession == "GSE10072":
        labels = ["tumor" if title.lower().startswith("lung tumor") else "non_tumor" for title in titles]
        groups = [title.rsplit("_", 1)[-1] for title in titles]
        return labels, groups
    raise ValueError(f"unsupported GEO accession: {accession}")


def prepare_geo_dataset(
    accession: str,
    raw_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    accession = accession.upper()
    spec = GEO_SPECS[accession]
    raw_dir = Path(raw_dir)
    ncbi_path = raw_dir / f"{accession}_series_matrix.txt.gz"
    if accession in {"GSE15471", "GSE10072"} and not ncbi_path.exists():
        feature_by_sample, metadata, sources = _ebi_fallback(accession, raw_dir)
    else:
        raw_path, source_url = download_series_matrix(accession, raw_dir)
        metadata = read_geo_metadata(raw_path)
        # GEO metadata/end markers start with ! and are safely ignored as comments.
        feature_by_sample = pd.read_csv(
            raw_path,
            sep="\t",
            comment="!",
            index_col=0,
            compression="gzip",
            low_memory=False,
        )
        sources = [(raw_path, source_url)]
    labels, groups = labels_and_groups(accession, metadata)
    sample_by_feature = feature_by_sample.T
    sample_by_feature.columns = sample_by_feature.columns.astype(str)
    if sample_by_feature.shape != (spec.expected_samples, spec.expected_features):
        raise ValueError(
            f"{accession} shape mismatch: {sample_by_feature.shape}, expected "
            f"({spec.expected_samples}, {spec.expected_features})"
        )
    if len(labels) != spec.expected_samples or len(set(labels)) != spec.expected_classes:
        raise ValueError(f"{accession} label integrity check failed")
    if spec.grouped and (groups is None or len(groups) != spec.expected_samples):
        raise ValueError(f"{accession} group integrity check failed")
    sample_by_feature["label"] = labels
    if groups is not None:
        sample_by_feature["group"] = groups
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.public_name}.csv.gz"
    sample_by_feature.to_csv(output_path, index=False, compression="gzip")
    record: dict[str, object] = {
        "spec": asdict(spec),
        "sources": [
            {"url": url, "path": str(path.resolve()), "sha256": sha256(path)}
            for path, url in sources
        ],
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "shape": {"samples": sample_by_feature.shape[0], "features": spec.expected_features},
        "class_counts": sample_by_feature["label"].value_counts().sort_index().to_dict(),
        "group_count": int(sample_by_feature["group"].nunique()) if groups is not None else None,
        "missing_expression_values": int(feature_by_sample.isna().sum().sum()),
    }
    (output_dir / f"{spec.public_name}.manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def _decode_labels(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value))


def _read_legacy(path: Path, spec: LegacySpec) -> tuple[pd.DataFrame, pd.Series]:
    if path.suffix.lower() == ".arff":
        records, _ = arff.loadarff(path)
        table = pd.DataFrame(records)
        features = table.iloc[:, :-1].apply(pd.to_numeric, errors="raise")
        labels = _decode_labels(table.iloc[:, -1])
        return features, labels
    if path.suffix.lower() == ".xlsx":
        # This source has no header: first column is the class, remaining columns are genes.
        table = pd.read_excel(path, header=None)
        return table.iloc[:, 1:].apply(pd.to_numeric, errors="raise"), _decode_labels(table.iloc[:, 0])
    if path.name.endswith(".tar.gz"):
        stem = "alon" if spec.public_name == "Colon" else "golub"
        with tarfile.open(path, "r:gz") as archive:
            input_handle = archive.extractfile(f"{stem}_inputs.csv")
            output_handle = archive.extractfile(f"{stem}_outputs.csv")
            if input_handle is None or output_handle is None:
                raise ValueError(f"missing matrices in {path.name}")
            features = pd.read_csv(input_handle, header=None)
            labels = _decode_labels(pd.read_csv(output_handle, header=None).iloc[:, 0])
        return features, labels
    raise ValueError(f"unsupported legacy data format: {path.name}")


def prepare_legacy_dataset(
    name: str,
    raw_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    if name not in LEGACY_SPECS:
        raise ValueError(f"unsupported legacy dataset: {name}")
    spec = LEGACY_SPECS[name]
    path = _download_file(spec.url, Path(raw_dir) / spec.filename)
    actual_hash = sha256(path)
    if actual_hash != spec.sha256:
        raise ValueError(f"{name} checksum mismatch: {actual_hash}")
    features, labels = _read_legacy(path, spec)
    if features.shape != (spec.expected_samples, spec.expected_features):
        raise ValueError(
            f"{name} shape mismatch: {features.shape}, expected "
            f"({spec.expected_samples}, {spec.expected_features})"
        )
    if labels.nunique() != spec.expected_classes or labels.isna().any():
        raise ValueError(f"{name} label integrity check failed")
    if int(features.isna().sum().sum()):
        raise ValueError(f"{name} contains missing expression values")
    features.columns = [f"feature_{index:05d}" for index in range(spec.expected_features)]
    output = features.copy()
    output["label"] = labels.to_numpy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.csv.gz"
    output.to_csv(output_path, index=False, compression="gzip")
    record: dict[str, object] = {
        "spec": asdict(spec),
        "source": {"url": spec.url, "path": str(path.resolve()), "sha256": actual_hash},
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "shape": {"samples": features.shape[0], "features": features.shape[1]},
        "class_counts": labels.value_counts().sort_index().to_dict(),
    }
    (output_dir / f"{name}.manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record

