# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests


def download_dataset(
    dataset_name: str,
    download_root: os.PathLike | str,
    download_url: str,
    *,
    verbose: bool = True,
) -> Path:
    """
    Download and extract a dataset archive (tar.gz/zip) or download a single csv/txt file.

    Returns:
        Path to the downloaded file (archive or csv/txt) in `download_root`.

    Raises:
        ValueError: if download_url is missing/empty.
        RuntimeError: if download/extract fails.
    """
    if not download_url or not isinstance(download_url, str):
        raise ValueError(
            f"{dataset_name} dataset cannot be automatically downloaded. Please download it manually."
        )

    root = Path(download_root)
    root.mkdir(parents=True, exist_ok=True)

    url_basename = Path(download_url.rstrip("/")).name
    is_plain_file = _is_csv_or_txt(download_url)
    is_zip_file = _is_zip(download_url)

    if is_plain_file or is_zip_file:
        filename = url_basename or ("dataset.zip" if is_zip_file else "dataset.csv")
        remove_finished = not is_plain_file  # keep csv/txt; remove archives by default
    else:
        filename = "archive.tar.gz"
        remove_finished = True

    dest_path = root / filename

    if verbose:
        print(f"Downloading '{dataset_name}' to {root}...")

    start = time.time()
    try:
        download_and_extract_archive(
            url=download_url,
            download_root=root,
            filename=filename,
            remove_finished=remove_finished,
            verbose=verbose,
        )
    except Exception as e:
        msg = (
            f"{dest_path} may be corrupted or incomplete. "
            f"Try deleting it and rerunning.\nOriginal error: {e!r}"
        )
        if verbose:
            print("\n" + msg + "\n")
        raise RuntimeError(msg) from e
    finally:
        if verbose:
            minutes = (time.time() - start) / 60
            print(f"\nDone in {minutes:.2f} minutes.\n")

    return dest_path


# -------------------------
# Helpers
# -------------------------


def _is_csv_or_txt(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".csv") or path.endswith(".txt")


def _is_zip(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".zip")


def _is_tar_gz(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".tar.gz") or path.endswith(".tgz")


def download_and_extract_archive(
    *,
    url: str,
    download_root: os.PathLike | str,
    filename: Optional[str] = None,
    remove_finished: bool = True,
    verbose: bool = True,
    chunk_size: int = 1024 * 1024,
    timeout_s: int = 60,
) -> Path:
    """
    Download `url` into `download_root/filename` and extract if it's a zip or tar.gz/tgz.
    If it's a plain file (e.g. csv/txt), it will just be downloaded.

    Returns:
        Path to the downloaded file.
    """
    root = Path(download_root)
    root.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = Path(urlparse(url).path).name or "download"

    archive_path = root / filename

    _download_url(
        url=url,
        dst=archive_path,
        verbose=verbose,
        chunk_size=chunk_size,
        timeout_s=timeout_s,
    )

    # Extract if archive
    if _is_zip(str(archive_path)) or archive_path.suffix.lower() == ".zip":
        _extract_zip(archive_path, root)
        if remove_finished:
            archive_path.unlink(missing_ok=True)
    elif _is_tar_gz(str(archive_path)) or archive_path.name.lower().endswith(
        (".tar.gz", ".tgz")
    ):
        _extract_tar_gz(archive_path, root)
        if remove_finished:
            archive_path.unlink(missing_ok=True)

    return archive_path


def _download_url(
    *,
    url: str,
    dst: Path,
    verbose: bool,
    chunk_size: int,
    timeout_s: int,
) -> None:
    """
    Stream-download `url` to `dst` with an atomic write to avoid partial files on failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")

    if verbose:
        print(f"Downloading {url} -> {dst}")

    try:
        with requests.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)

            written = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)

                    if verbose and total > 0:
                        pct = (written / total) * 100
                        print(f"\r{pct:6.2f}% ({written}/{total} bytes)", end="")
        if verbose and total > 0:
            print()  # newline after progress

        tmp.replace(dst)
    except Exception:
        # Best-effort cleanup of partial download
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _extract_zip(zip_path: Path, dst_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extract_zip(zf, dst_dir)


def _extract_tar_gz(tar_path: Path, dst_dir: Path) -> None:
    mode = "r:gz" if tar_path.name.lower().endswith((".tar.gz", ".tgz")) else "r:*"
    with tarfile.open(tar_path, mode) as tf:
        _safe_extract_tar(tf, dst_dir)


def _safe_extract_zip(zf: zipfile.ZipFile, dst_dir: Path) -> None:
    dst_dir = dst_dir.resolve()
    for member in zf.infolist():
        member_path = (dst_dir / member.filename).resolve()
        if not str(member_path).startswith(str(dst_dir) + os.sep):
            raise RuntimeError(f"Unsafe zip path traversal attempt: {member.filename}")
    zf.extractall(dst_dir)


def _safe_extract_tar(tf: tarfile.TarFile, dst_dir: Path) -> None:
    dst_dir = dst_dir.resolve()
    for member in tf.getmembers():
        member_path = (dst_dir / member.name).resolve()
        if not str(member_path).startswith(str(dst_dir) + os.sep):
            raise RuntimeError(f"Unsafe tar path traversal attempt: {member.name}")
    tf.extractall(dst_dir)
