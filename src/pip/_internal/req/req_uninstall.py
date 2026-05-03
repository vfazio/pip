from __future__ import annotations

import functools
import os
import stat
import sys
import sysconfig
from collections.abc import Callable, Generator, Iterable
from importlib.util import cache_from_source
from typing import Any

from pip._internal.exceptions import LegacyDistutilsInstall, UninstallMissingRecord
from pip._internal.locations import get_bin_prefix, get_bin_user
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.logging import getLogger, indent_log
from pip._internal.utils.misc import ask, normalize_path, renames, rmtree
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory
from pip._internal.utils.virtualenv import running_under_virtualenv

logger = getLogger(__name__)


def _script_names(
    bin_dir: str, script_name: str, is_gui: bool
) -> Generator[str, None, None]:
    """Create the fully qualified name of the files created by
    {console,gui}_scripts for the given ``dist``.
    Returns the list of file names
    """
    exe_name = os.path.join(bin_dir, script_name)
    yield exe_name
    if not WINDOWS:
        return
    yield f"{exe_name}.exe"
    yield f"{exe_name}.exe.manifest"
    if is_gui:
        yield f"{exe_name}-script.pyw"
    else:
        yield f"{exe_name}-script.py"


def _unique(
    fn: Callable[..., Generator[Any, None, None]],
) -> Callable[..., Generator[Any, None, None]]:
    @functools.wraps(fn)
    def unique(*args: Any, **kw: Any) -> Generator[Any, None, None]:
        seen: set[Any] = set()
        for item in fn(*args, **kw):
            if item not in seen:
                seen.add(item)
                yield item

    return unique


@_unique
def uninstallation_paths(dist: BaseDistribution) -> Generator[str, None, None]:
    """
    Yield all the uninstallation paths for dist based on RECORD-without-.py[co]

    Yield paths to all the files in RECORD. For each .py file in RECORD, add
    the .pyc and .pyo in the same directory.

    UninstallPathSet.add() takes care of the __pycache__ .py[co].

    If RECORD is not found, raises an error,
    with possible information from the INSTALLER file.

    https://packaging.python.org/specifications/recording-installed-packages/
    """
    location = dist.location
    assert location is not None, "not installed"

    entries = dist.iter_declared_entries()
    if entries is None:
        raise UninstallMissingRecord(distribution=dist)

    for entry in entries:
        path = os.path.join(location, entry)
        yield path
        if path.endswith(".py"):
            dn, fn = os.path.split(path)
            base = fn[:-3]
            path = os.path.join(dn, base + ".pyc")
            yield path
            path = os.path.join(dn, base + ".pyo")
            yield path


def compact(paths: Iterable[str]) -> set[str]:
    """Compact a path set to contain the minimal number of paths
    necessary to contain all paths in the set. If /a/path/ and
    /a/path/to/a/file.txt are both in the set, leave only the
    shorter path."""

    sep = os.path.sep
    short_paths: set[str] = set()
    for path in sorted(paths, key=len):
        should_skip = any(
            path.startswith(shortpath.rstrip("*"))
            and path[len(shortpath.rstrip("*").rstrip(sep))] == sep
            for shortpath in short_paths
        )
        if not should_skip:
            short_paths.add(path)
    return short_paths


def compress_for_rename(paths: Iterable[str]) -> set[str]:
    """Returns a set containing the paths that need to be renamed.

    This set may include directories when the original sequence of paths
    included every file on disk.

    Note that since we do not control the input into this function, we do things
    like checking for os.sep in case external logic ever changes.
    """
    case_map: dict[str, str] = {}
    remaining: set[str] = set()
    wildcards: dict[str, str] = {}

    def norm_join(*a: str) -> str:
        return os.path.normcase(os.path.join(*a))

    # Map normalized roots to their original casing
    potential_roots: dict[str, str] = {}
    covered_cache: dict[str, bool] = {}

    for path in sorted(paths, key=len):
        norm_path = os.path.normcase(path)
        norm_dir = os.path.dirname(norm_path)

        # We do _not_ add the files within wildcard paths.
        # To speed up whether a file is within a wildcard, we break the
        # path up into components and cache the answer so other files can
        # take advantage of the lookup.
        is_covered = covered_cache.get(norm_dir)
        if is_covered is None:
            # Only do the expensive walk if the cache misses
            curr = norm_dir
            is_covered = False
            while curr:
                w_key = curr if curr.endswith(os.sep) else curr + os.sep
                if w_key in wildcards:
                    is_covered = True
                    break
                parent = os.path.dirname(curr)
                if parent == curr:
                    break
                curr = parent
            covered_cache[norm_dir] = is_covered

        if is_covered:
            continue

        try:
            if stat.S_ISDIR(os.stat(path, follow_symlinks=False).st_mode):
                w_key = norm_path if norm_path.endswith(os.sep) else norm_path + os.sep
                w_orig = path if path.endswith(os.sep) else path + os.sep
                wildcards[w_key] = w_orig
                covered_cache[norm_path] = True
                continue
        except OSError:
            continue

        case_map[norm_path] = path
        remaining.add(norm_path)
        p_dir_norm = norm_dir if norm_dir.endswith(os.sep) else norm_dir + os.sep
        if p_dir_norm not in potential_roots:
            orig_dir = os.path.dirname(path)
            p_dir_orig = orig_dir if orig_dir.endswith(os.sep) else orig_dir + os.sep
            potential_roots[p_dir_norm] = p_dir_orig

    # Outside of the initial pass, all data should be in a known format

    # We want to identify the top most unique candidate directories so that we
    # only process a directory and its children once
    roots: list[str] = []
    for candidate in sorted(potential_roots, key=len):
        if not any(candidate.startswith(os.path.normcase(r)) for r in roots):
            roots.append(potential_roots[candidate])

    # a list of all directories we may own
    owned_paths: set[str] = set()
    for rs_norm in potential_roots:
        for r_orig in roots:
            r_norm = os.path.normcase(r_orig)

            if rs_norm.startswith(r_norm):
                # Calculate the lineage segments
                tail = rs_norm[len(r_norm) :]

                current = r_norm
                owned_paths.add(current)

                if tail:
                    parts = [p for p in tail.split(os.sep) if p]
                    for part in parts:
                        current = current + part + os.sep
                        owned_paths.add(current)
                break

    def process_directory(real_dir: str) -> bool:
        """
        Returns True if the directory is perfectly clean.
        Reads the disk exactly once per valid directory.

        real_dir should generally not be in normcase for purposes of fidelity
        """
        norm_dir = norm_join(real_dir, "")

        if norm_dir in wildcards:
            return True

        try:
            with os.scandir(real_dir) as it:
                entries = list(it)
        except OSError:
            return False

        poisoned = False
        local_files = set()

        for entry in entries:
            norm_entry = os.path.normcase(entry.path)

            if entry.is_dir(follow_symlinks=False):
                _dir = norm_entry + os.sep

                # is the dir in our wildcards?
                if _dir in wildcards:
                    continue

                # Is the path out of bounds? A possible scenario is a sibling
                # directory for a namespace package not owned by the distribution
                # being uninstalled. If so, we cannot wildcard the current
                # directory and we do not need to process files in that path.
                # We must continue processing this path for files we may still own
                #
                # Note: Even if the candidate dir is empty, we do _not_ remove it.
                # To do so would require processing the directory and for it to be
                # clean (not poisoned), we could then possibly clean up the dir.
                #
                # The packaging guidelines say:
                #   Directories should not be listed.
                # and
                #   To completely uninstall a package, a tool needs to remove all
                #   files listed in RECORD, all .pyc files (of all optimization
                #   levels) corresponding to removed .py files, _and any directories
                #   emptied by the uninstallation_.
                # See:
                #   https://packaging.python.org/en/latest/specifications/recording-installed-packages/#the-record-file
                #
                # Since this RECORD did not own that directory, and should != shall,
                # meaning some package could have made that empty path, we're
                # following the guidance to the letter.
                # If we change our mind:
                #   poisoned = not process_directory(entry.path)
                if _dir not in owned_paths:
                    poisoned = True
                    continue

                # if the descendant is not wholly captured by the file list and
                # thus a wildcard, then we cannot be a wildcard either.
                if not process_directory(entry.path):
                    poisoned = True
            else:
                # We've identified a file in the current directory and have no directive
                # to claim is as our own (wildcard or discrete entry in remaining)
                if norm_entry in remaining:
                    local_files.add(norm_entry)
                else:
                    poisoned = True

        if poisoned:
            return False

        # If we claim ownership of all physical files/subdirectories via wildcards
        # or entries in the file list, then we are a wildcard and can remove all
        # entries matching our prefix.
        wildcards[norm_dir] = os.path.join(real_dir, "")
        remaining.difference_update(local_files)
        # This covers the case where a file is in the file list but not on disk and
        # we never iterated on it. Remove so there are no rogue entries in `remaining`.
        # We could improve this with a mapping of parent path to files
        remaining.difference_update({f for f in remaining if f.startswith(norm_dir)})
        return True

    # Process the top level roots we identified
    for root in roots:
        process_directory(root)

    # Because a child adds itself to wildcards before its parent does,
    # we need to filter out the children if the parent ultimately succeeded.
    final_wildcards: set[str] = set()
    for w in sorted(wildcards, key=len):
        orig_w = wildcards[w]
        if not any(w.startswith(os.path.normcase(fw)) for fw in final_wildcards):
            final_wildcards.add(orig_w)

    return {case_map[p] for p in remaining} | final_wildcards


def compress_for_output_listing(paths: Iterable[str]) -> tuple[set[str], set[str]]:
    """Returns a tuple of 2 sets of which paths to display to user

    The first set contains paths that would be deleted. Files of a package
    are not added and the top-level directory of the package has a '*' added
    at the end - to signify that all it's contents are removed.

    The second set contains files that would have been skipped in the above
    folders.
    """

    will_remove = set(paths)
    will_skip = set()

    # Determine folders and files
    folders = set()
    files = set()
    for path in will_remove:
        if path.endswith(".pyc"):
            continue
        if path.endswith("__init__.py") or ".dist-info" in path:
            folders.add(os.path.dirname(path))
        files.add(path)

    _normcased_files = set(map(os.path.normcase, files))

    _scheduled_dirs = {
        os.path.normcase(p)
        for p in will_remove
        if os.path.isdir(p) and not os.path.islink(p)
    }

    folders = compact(folders)

    # This walks the tree using os.walk to not miss extra folders
    # that might get added.
    for folder in folders:
        for dirpath, subdirs, dirfiles in os.walk(folder):
            subdirs[:] = [
                d
                for d in subdirs
                if os.path.normcase(os.path.join(dirpath, d)) not in _scheduled_dirs
            ]

            for fname in dirfiles:
                if fname.endswith(".pyc"):
                    continue

                file_ = os.path.join(dirpath, fname)
                if (
                    os.path.isfile(file_)
                    and os.path.normcase(file_) not in _normcased_files
                ):
                    # We are skipping this file. Add it to the set.
                    will_skip.add(file_)

    will_remove = files | {os.path.join(folder, "*") for folder in folders}

    return will_remove, will_skip


class StashedUninstallPathSet:
    """A set of file rename operations to stash files while
    tentatively uninstalling them."""

    def __init__(self) -> None:
        # Mapping from source file root to [Adjacent]TempDirectory
        # for files under that directory.
        self._save_dirs: dict[str, TempDirectory] = {}
        # (old path, new path) tuples for each move that may need
        # to be undone.
        self._moves: list[tuple[str, str]] = []

    def _get_directory_stash(self, path: str) -> str:
        """Stashes a directory.

        Directories are stashed adjacent to their original location if
        possible, or else moved/copied into the user's temp dir."""

        try:
            save_dir: TempDirectory = AdjacentTempDirectory(path)
        except OSError:
            save_dir = TempDirectory(kind="uninstall")
        self._save_dirs[os.path.normcase(path)] = save_dir

        return save_dir.path

    def _get_file_stash(self, path: str) -> str:
        """Stashes a file.

        If no root has been provided, one will be created for the directory
        in the user's temp directory."""
        path = os.path.normcase(path)
        head, old_head = os.path.dirname(path), None
        save_dir = None

        while head != old_head:
            try:
                save_dir = self._save_dirs[head]
                break
            except KeyError:
                pass
            head, old_head = os.path.dirname(head), head
        else:
            # Did not find any suitable root
            head = os.path.dirname(path)
            save_dir = TempDirectory(kind="uninstall")
            self._save_dirs[head] = save_dir

        relpath = os.path.relpath(path, head)
        if relpath and relpath != os.path.curdir:
            return os.path.join(save_dir.path, relpath)
        return save_dir.path

    def stash(self, path: str) -> str:
        """Stashes the directory or file and returns its new location.
        Handle symlinks as files to avoid modifying the symlink targets.
        """
        path_is_dir = os.path.isdir(path) and not os.path.islink(path)
        if path_is_dir:
            new_path = self._get_directory_stash(path)
        else:
            new_path = self._get_file_stash(path)

        self._moves.append((path, new_path))
        if path_is_dir and os.path.isdir(new_path):
            # If we're moving a directory, we need to
            # remove the destination first or else it will be
            # moved to inside the existing directory.
            # We just created new_path ourselves, so it will
            # be removable.
            os.rmdir(new_path)
        renames(path, new_path)
        return new_path

    def commit(self) -> None:
        """Commits the uninstall by removing stashed files."""
        for save_dir in self._save_dirs.values():
            save_dir.cleanup()
        self._moves = []
        self._save_dirs = {}

    def rollback(self) -> None:
        """Undoes the uninstall by moving stashed files back."""
        for p in self._moves:
            logger.info("Moving to %s\n from %s", *p)

        for new_path, path in self._moves:
            try:
                logger.debug("Replacing %s from %s", new_path, path)
                if os.path.isfile(new_path) or os.path.islink(new_path):
                    os.unlink(new_path)
                elif os.path.isdir(new_path):
                    rmtree(new_path)
                renames(path, new_path)
            except OSError as ex:
                logger.error("Failed to restore %s", new_path)
                logger.debug("Exception: %s", ex)

        self.commit()

    @property
    def can_rollback(self) -> bool:
        return bool(self._moves)


class UninstallPathSet:
    """A set of file paths to be removed in the uninstallation of a
    requirement."""

    def __init__(self, dist: BaseDistribution) -> None:
        self._paths: set[str] = set()
        self._refuse: set[str] = set()
        self._pth: dict[str, UninstallPthEntries] = {}
        self._dist = dist
        self._moved_paths = StashedUninstallPathSet()
        # Create local cache of normalize_path results. Creating an UninstallPathSet
        # can result in hundreds/thousands of redundant calls to normalize_path with
        # the same args, which hurts performance.
        self._normalize_path_cached = functools.lru_cache(normalize_path)

    def _permitted(self, path: str) -> bool:
        """
        Return True if the given path is one we are permitted to
        remove/modify, False otherwise.

        """
        # aka is_local, but caching normalized sys.prefix
        if not running_under_virtualenv():
            return True
        return path.startswith(self._normalize_path_cached(sys.prefix))

    def add(self, path: str) -> None:
        head, tail = os.path.split(path)

        # we normalize the head to resolve parent directory symlinks, but not
        # the tail, since we only want to uninstall symlinks, not their targets
        path = os.path.join(self._normalize_path_cached(head), os.path.normcase(tail))

        if not os.path.exists(path):
            return
        if self._permitted(path):
            self._paths.add(path)
        else:
            self._refuse.add(path)

        # __pycache__ files can show up after 'installed-files.txt' is created,
        # due to imports
        if os.path.splitext(path)[1] == ".py":
            self.add(cache_from_source(path))

    def add_pth(self, pth_file: str, entry: str) -> None:
        pth_file = self._normalize_path_cached(pth_file)
        if self._permitted(pth_file):
            if pth_file not in self._pth:
                self._pth[pth_file] = UninstallPthEntries(pth_file)
            self._pth[pth_file].add(entry)
        else:
            self._refuse.add(pth_file)

    def remove(self, auto_confirm: bool = False, verbose: bool = False) -> None:
        """Remove paths in ``self._paths`` with confirmation (unless
        ``auto_confirm`` is True)."""

        if not self._paths:
            logger.info(
                "Can't uninstall '%s'. No files were found to uninstall.",
                self._dist.raw_name,
            )
            return

        dist_name_version = f"{self._dist.raw_name}-{self._dist.raw_version}"
        logger.info("Uninstalling %s:", dist_name_version)

        with indent_log():
            if auto_confirm or self._allowed_to_proceed(verbose):
                moved = self._moved_paths

                for_rename = compress_for_rename(self._paths)

                for path in sorted(compact(for_rename)):
                    moved.stash(path)
                    logger.verbose("Removing file or directory %s", path)

                for pth in self._pth.values():
                    pth.remove()

                logger.info("Successfully uninstalled %s", dist_name_version)

    def _allowed_to_proceed(self, verbose: bool) -> bool:
        """Display which files would be deleted and prompt for confirmation"""

        def _display(msg: str, paths: Iterable[str]) -> None:
            if not paths:
                return

            logger.info(msg)
            with indent_log():
                for path in sorted(compact(paths)):
                    logger.info(path)

        if not verbose:
            will_remove, will_skip = compress_for_output_listing(self._paths)
        else:
            # In verbose mode, display all the files that are going to be
            # deleted.
            will_remove = set(self._paths)
            will_skip = set()

        _display("Would remove:", will_remove)
        _display("Would not remove (might be manually added):", will_skip)
        _display("Would not remove (outside of prefix):", self._refuse)
        if verbose:
            _display("Will actually move:", compress_for_rename(self._paths))

        return ask("Proceed (Y/n)? ", ("y", "n", "")) != "n"

    def rollback(self) -> None:
        """Rollback the changes previously made by remove()."""
        if not self._moved_paths.can_rollback:
            logger.error(
                "Can't roll back %s; was not uninstalled",
                self._dist.raw_name,
            )
            return
        logger.info("Rolling back uninstall of %s", self._dist.raw_name)
        self._moved_paths.rollback()
        for pth in self._pth.values():
            pth.rollback()

    def commit(self) -> None:
        """Remove temporary save dir: rollback will no longer be possible."""
        self._moved_paths.commit()

    @classmethod
    def from_dist(cls, dist: BaseDistribution) -> UninstallPathSet:
        dist_location = dist.location
        info_location = dist.info_location
        if dist_location is None:
            logger.info(
                "Not uninstalling %s since it is not installed",
                dist.canonical_name,
            )
            return cls(dist)

        normalized_dist_location = normalize_path(dist_location)
        if not dist.local:
            logger.info(
                "Not uninstalling %s at %s, outside environment %s",
                dist.canonical_name,
                normalized_dist_location,
                sys.prefix,
            )
            return cls(dist)

        if normalized_dist_location in {
            p
            for p in {sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")}
            if p
        }:
            logger.info(
                "Not uninstalling %s at %s, as it is in the standard library.",
                dist.canonical_name,
                normalized_dist_location,
            )
            return cls(dist)

        paths_to_remove = cls(dist)
        develop_egg_link = egg_link_path_from_location(dist.raw_name)

        # Distribution is installed with metadata in a "flat" .egg-info
        # directory. This means it is not a modern .dist-info installation, an
        # egg, or legacy editable.
        setuptools_flat_installation = (
            dist.installed_with_setuptools_egg_info
            and info_location is not None
            and os.path.exists(info_location)
            # If dist is editable and the location points to a ``.egg-info``,
            # we are in fact in the legacy editable case.
            and not info_location.endswith(f"{dist.setuptools_filename}.egg-info")
        )

        # Uninstall cases order do matter as in the case of 2 installs of the
        # same package, pip needs to uninstall the currently detected version
        if setuptools_flat_installation:
            if info_location is not None:
                paths_to_remove.add(info_location)
            installed_files = dist.iter_declared_entries()
            if installed_files is not None:
                for installed_file in installed_files:
                    paths_to_remove.add(os.path.join(dist_location, installed_file))
            # FIXME: need a test for this elif block
            # occurs with --single-version-externally-managed/--record outside
            # of pip
            elif dist.is_file("top_level.txt"):
                try:
                    namespace_packages = dist.read_text("namespace_packages.txt")
                except FileNotFoundError:
                    namespaces = []
                else:
                    namespaces = namespace_packages.splitlines(keepends=False)
                for top_level_pkg in [
                    p
                    for p in dist.read_text("top_level.txt").splitlines()
                    if p and p not in namespaces
                ]:
                    path = os.path.join(dist_location, top_level_pkg)
                    paths_to_remove.add(path)
                    paths_to_remove.add(f"{path}.py")
                    paths_to_remove.add(f"{path}.pyc")
                    paths_to_remove.add(f"{path}.pyo")

        elif dist.installed_by_distutils:
            raise LegacyDistutilsInstall(distribution=dist)

        elif dist.installed_as_egg:
            # package installed by easy_install
            # We cannot match on dist.egg_name because it can slightly vary
            # i.e. setuptools-0.6c11-py2.6.egg vs setuptools-0.6rc11-py2.6.egg
            # XXX We use normalized_dist_location because dist_location my contain
            # a trailing / if the distribution is a zipped egg
            # (which is not a directory).
            paths_to_remove.add(normalized_dist_location)
            easy_install_egg = os.path.split(normalized_dist_location)[1]
            easy_install_pth = os.path.join(
                os.path.dirname(normalized_dist_location),
                "easy-install.pth",
            )
            paths_to_remove.add_pth(easy_install_pth, "./" + easy_install_egg)

        elif dist.installed_with_dist_info:
            for path in uninstallation_paths(dist):
                paths_to_remove.add(path)

        elif develop_egg_link:
            # PEP 660 modern editable is handled in the ``.dist-info`` case
            # above, so this only covers the setuptools-style editable.
            with open(develop_egg_link) as fh:
                link_pointer = os.path.normcase(fh.readline().strip())
                normalized_link_pointer = paths_to_remove._normalize_path_cached(
                    link_pointer
                )
            assert os.path.samefile(
                normalized_link_pointer, normalized_dist_location
            ), (
                f"Egg-link {develop_egg_link} (to {link_pointer}) does not match "
                f"installed location of {dist.raw_name} (at {dist_location})"
            )
            paths_to_remove.add(develop_egg_link)
            easy_install_pth = os.path.join(
                os.path.dirname(develop_egg_link), "easy-install.pth"
            )
            paths_to_remove.add_pth(easy_install_pth, dist_location)

        else:
            logger.debug(
                "Not sure how to uninstall: %s - Check: %s",
                dist,
                dist_location,
            )

        if dist.in_usersite:
            bin_dir = get_bin_user()
        else:
            bin_dir = get_bin_prefix()

        # find distutils scripts= scripts
        try:
            for script in dist.iter_distutils_script_names():
                paths_to_remove.add(os.path.join(bin_dir, script))
                if WINDOWS:
                    paths_to_remove.add(os.path.join(bin_dir, f"{script}.bat"))
        except (FileNotFoundError, NotADirectoryError):
            pass

        # find console_scripts and gui_scripts
        def iter_scripts_to_remove(
            dist: BaseDistribution,
            bin_dir: str,
        ) -> Generator[str, None, None]:
            for entry_point in dist.iter_entry_points():
                if entry_point.group == "console_scripts":
                    yield from _script_names(bin_dir, entry_point.name, False)
                elif entry_point.group == "gui_scripts":
                    yield from _script_names(bin_dir, entry_point.name, True)

        for s in iter_scripts_to_remove(dist, bin_dir):
            paths_to_remove.add(s)

        return paths_to_remove


class UninstallPthEntries:
    def __init__(self, pth_file: str) -> None:
        self.file = pth_file
        self.entries: set[str] = set()
        self._saved_lines: list[bytes] | None = None

    def add(self, entry: str) -> None:
        entry = os.path.normcase(entry)
        # On Windows, os.path.normcase converts the entry to use
        # backslashes.  This is correct for entries that describe absolute
        # paths outside of site-packages, but all the others use forward
        # slashes.
        # os.path.splitdrive is used instead of os.path.isabs because isabs
        # treats non-absolute paths with drive letter markings like c:foo\bar
        # as absolute paths. It also does not recognize UNC paths if they don't
        # have more than "\\sever\share". Valid examples: "\\server\share\" or
        # "\\server\share\folder".
        if WINDOWS and not os.path.splitdrive(entry)[0]:
            entry = entry.replace("\\", "/")
        self.entries.add(entry)

    def remove(self) -> None:
        logger.verbose("Removing pth entries from %s:", self.file)

        # If the file doesn't exist, log a warning and return
        if not os.path.isfile(self.file):
            logger.warning("Cannot remove entries from nonexistent file %s", self.file)
            return
        with open(self.file, "rb") as fh:
            # windows uses '\r\n' with py3k, but uses '\n' with py2.x
            lines = fh.readlines()
            self._saved_lines = lines
        if any(b"\r\n" in line for line in lines):
            endline = "\r\n"
        else:
            endline = "\n"
        # handle missing trailing newline
        if lines and not lines[-1].endswith(endline.encode("utf-8")):
            lines[-1] = lines[-1] + endline.encode("utf-8")
        for entry in self.entries:
            try:
                logger.verbose("Removing entry: %s", entry)
                lines.remove((entry + endline).encode("utf-8"))
            except ValueError:
                pass
        with open(self.file, "wb") as fh:
            fh.writelines(lines)

    def rollback(self) -> bool:
        if self._saved_lines is None:
            logger.error("Cannot roll back changes to %s, none were made", self.file)
            return False
        logger.debug("Rolling %s back to previous state", self.file)
        with open(self.file, "wb") as fh:
            fh.writelines(self._saved_lines)
        return True
