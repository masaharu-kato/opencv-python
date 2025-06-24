import random
from pathlib import Path
from collections.abc import Iterable
from typing import IO

class PathGroup:
    """A group of paths representing a good frame and multiple bad frames."""

    def __init__(self, gpath: Path, bpaths: list[Path]):
        self._gpath = gpath # Path to the good frame image
        self._bpaths = bpaths # Paths to the bad frame images

    @classmethod
    def load(cls, gpath: Path | str, bpaths: Iterable[Path | str]):
        return cls(Path(gpath), [Path(bpath) for bpath in bpaths])
    
    @classmethod
    def load_from_file(cls, line: str, root_dir: Path) -> 'PathGroup':
        """Loads a PathGroup from a line in the format: good_path\tbad_path1\tbad_path2\t..."""
        paths = [root_dir / relpath for relpath in line.strip().split('\t')]
        if len(paths) < 2:
            raise ValueError(f"Invalid line format: {line}")
        return cls(paths[0], paths[1:])

    def iter_pairs(self) -> Iterable[tuple[Path, Path]]:
        """Returns an iterable of (good, bad) image relative path pairs."""
        for bpath in self._bpaths:
            yield self._gpath, bpath

    def dump(self) -> tuple[str, list[str]]:
        """Returns a tuple of (good path, list of bad paths) for serialization."""
        return str(self._gpath), [str(bpath) for bpath in self._bpaths]
    
    def dump_to_file(self, fw: IO, root_dir: Path):
        """Dumps the group to a file in the format: good_path\tbad_path1\tbad_path2\t..."""
        print("\t".join((str(path.relative_to(root_dir)) for path in (self._gpath, *self._bpaths))), file=fw)


class PathGroups:
    """A collection of PathGroup objects representing good and bad image pairs."""

    def __init__(self, path_groups: list[PathGroup] | None = None):
        self.path_groups = path_groups if path_groups is not None else []

    def __len__(self) -> int:
        """Returns the number of PathGroups."""
        return len(self.path_groups)
    
    def __getitem__(self, idx: int) -> PathGroup:
        """Returns the PathGroup at the given index."""
        if idx < 0 or idx >= len(self.path_groups):
            raise IndexError("Index out of range.")
        return self.path_groups[idx]
        
    def __iter__(self):
        """Returns an iterator over the PathGroups."""
        for group in self.path_groups:
            yield group

    def copy(self) -> 'PathGroups':
        """Returns a shallow copy of the PathGroups."""
        return PathGroups(self.path_groups.copy())
    
    def copy_shuffled(self) -> 'PathGroups':
        """Returns a shuffled copy of the PathGroups."""
        shuffled_groups = self.path_groups.copy()
        random.shuffle(shuffled_groups)
        return PathGroups(shuffled_groups)
    
    def split(self, split_index: int) -> tuple['PathGroups', 'PathGroups']:
        """Splits the PathGroups into two groups at the given index."""
        if split_index < 0 or split_index > len(self.path_groups):
            raise ValueError("Split index out of range.")
        return PathGroups(self.path_groups[:split_index]), PathGroups(self.path_groups[split_index:])

    def extend(self, path_groups: Iterable[tuple[Path | str, Iterable[Path | str]]]):
        """Loads the dataset from a list of (good path, bad paths) tuples."""
        self.path_groups.extend([PathGroup.load(gpath, bpaths) for gpath, bpaths in path_groups])

    @classmethod
    def from_dump(cls, dump: list[tuple[str, list[str]]]) -> 'PathGroups':
        path_groups = cls()
        path_groups.extend(dump)
        return path_groups
    
    def extend_from_file(self, file_path: Path | str):
        """Loads the dataset from a file in the format: good_path\tbad_path1\tbad_path2\t..."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset file {file_path} does not exist.")
        with file_path.open('r') as fr:
            for line in fr:
                if line.startswith('#'):
                    continue  # Skip comments
                group = PathGroup.load_from_file(line, file_path.parent)
                self.path_groups.append(group)

    def append(self, group: PathGroup):
        """Appends a PathGroup to the collection."""
        if not isinstance(group, PathGroup):
            raise TypeError("Expected a PathGroup instance.")
        self.path_groups.append(group)
    
    def dump(self) -> list[tuple[str, list[str]]]:
        """Dumps the dataset to a list of (good path, list of bad paths) tuples for serialization."""
        return [group.dump() for group in self.path_groups]
    
    def dump_to_file(self, file_path: Path | str):
        """Dumps the dataset to a file in the format: good_path\tbad_path1\tbad_path2\t..."""
        file_path = Path(file_path)
        if file_path.is_dir():
            raise ValueError(f"Cannot dump dataset to a directory: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
        with file_path.open('w') as fw:
            for group in self.path_groups:
                group.dump_to_file(fw, file_path.parent)

