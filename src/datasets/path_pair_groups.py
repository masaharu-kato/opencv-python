from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import random
import re
import logging

@dataclass
class PathPair:
    """A pair of paths representing a good frame and a bad frame."""
    bpath: Path
    gpath: Path


class PathPairGroup:
    """A group of path pairs representing a good frame and multiple bad frames."""

    def __init__(self, pairs: list[PathPair] | None = None):
        self.pairs = pairs if pairs is not None else []

    def __len__(self) -> int:
        """Returns the number of pairs in the group."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PathPair:
        """Returns the PathPair at the given index."""
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError("Index out of range.")
        return self.pairs[idx]

    def __iter__(self):
        """Returns an iterator over the PathPairs."""
        for pair in self.pairs:
            yield pair

    def copy(self) -> 'PathPairGroup':
        """Returns a shallow copy of the PathPairGroup."""
        return PathPairGroup(self.pairs.copy())
    
    def append(self, pair: PathPair):
        """Appends a PathPair to the group."""
        self.pairs.append(pair)

    @classmethod
    def from_dump(cls, pairs: list[dict]) -> 'PathPairGroup':
        """Loads a PathPairGroup from a list of dictionaries."""
        return cls([PathPair(Path(pair['bpath']), Path(pair['gpath'])) for pair in pairs])


class PathPairGroups:
    """A collection of PathPairGroup objects representing good and bad image pairs."""

    def __init__(self, ppair_groups: list[PathPairGroup] | None = None):
        self.ppair_groups = ppair_groups if ppair_groups is not None else []

    def __len__(self) -> int:
        """Returns the number of PathGroups."""
        return len(self.ppair_groups)
    
    def __getitem__(self, idx: int) -> PathPairGroup:
        """Returns the PathGroup at the given index."""
        if idx < 0 or idx >= len(self.ppair_groups):
            raise IndexError("Index out of range.")
        return self.ppair_groups[idx]
        
    def __iter__(self):
        """Returns an iterator over the PathGroups."""
        for group in self.ppair_groups:
            yield group

    def append(self, group: PathPairGroup):
        """Appends a PathPairGroup to the collection."""
        self.ppair_groups.append(group)

    def extend(self, groups: Iterable[PathPairGroup]):
        """Extends the collection with multiple PathPairGroups."""
        self.ppair_groups.extend(groups)

    def copy(self) -> 'PathPairGroups':
        """Returns a shallow copy of the PathGroups."""
        return PathPairGroups(self.ppair_groups.copy())
    
    def copy_shuffled(self) -> 'PathPairGroups':
        """Returns a shuffled copy of the PathGroups."""
        shuffled_groups = self.ppair_groups.copy()
        random.shuffle(shuffled_groups)
        return PathPairGroups(shuffled_groups)
    
    def split_by_index(self, split_index: int) -> tuple['PathPairGroups', 'PathPairGroups']:
        """Splits the PathGroups into two groups at the given index."""
        if split_index < 0 or split_index > len(self.ppair_groups):
            raise ValueError("Split index out of range.")
        return PathPairGroups(self.ppair_groups[:split_index]), PathPairGroups(self.ppair_groups[split_index:])
    
    def split(self, ratio: float) -> tuple['PathPairGroups', 'PathPairGroups']:
        """Splits into two datasets baased on the image groups"""
        if len(self.ppair_groups) == 0:
            raise ValueError("Empty dataset, cannot split.")
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be between 0 and 1.")
        if len(self.ppair_groups) == 1:
            if ratio == 0:
                return PathPairGroups(self.ppair_groups), PathPairGroups()  # Return the original dataset and an empty one
            elif ratio == 1:
                return PathPairGroups(), PathPairGroups(self.ppair_groups)  # Return an empty dataset and the original one
            raise ValueError("Cannot split a single group dataset without 0 or 1 ratio.")
        
        split_index = min(max(1, int(len(self.ppair_groups) * ratio)), len(self.ppair_groups) - 1)  # Ensure at least one group in each split
        return self.split_by_index(split_index)
    
    def random_split(self, ratio: float) -> tuple['PathPairGroups', 'PathPairGroups']:
        """Randomly splits the dataset into two datasets based on the given ratio."""
        shuffled = self.copy_shuffled()  # Shuffle the groups before splitting
        return shuffled.split(ratio)
    
    def combined(self, other: 'PathPairGroups') -> 'PathPairGroups':
        """Combines this PathPairGroups with another one."""
        combined_groups = self.ppair_groups + other.ppair_groups
        return PathPairGroups(combined_groups)
    
    def dump(self) -> list[list[dict]]:
        """Returns a list of lists of dictionaries representing the path pairs."""
        return [[{'bpath': str(pair.bpath), 'gpath': str(pair.gpath)} for pair in group.pairs] for group in self.ppair_groups]

    @classmethod
    def from_dump(cls, _pairs_list: list[list[dict]]) -> 'PathPairGroups':
        ppair_groups = [PathPairGroup.from_dump(pairs) for pairs in _pairs_list]
        return cls(ppair_groups)
    

    _RE_FILENAME = re.compile(r"ig(?P<ig>\d+)_ib(?P<ib>\d+)_(?P<gorb>[gb]).png$")

    @classmethod
    def from_dir(cls, root_dir: Path) -> 'PathPairGroups':
        """Loads path pairs from a directory structure."""
        
        ppair_groups = cls()
        error = False

        for sub_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):

            groups_by_ig: dict[int, PathPairGroup] = defaultdict(lambda: PathPairGroup())
            last_bpath: Path | None = None

            for path in sorted((path for path in sub_dir.glob("*.png") if path.is_file())):

                # path = path.relative_to(root_dir)

                if m := cls._RE_FILENAME.match(path.name):

                    ig = int(m.group('ig'))
                    ib = int(m.group('ib'))
                    gorb = m.group('gorb')

                    if gorb == 'b':
                        if last_bpath is not None:
                            logging.error(f"Duplicate bad frame detected: {path} (previous: {last_bpath})")
                            error = True
                        last_bpath = path
                    else:
                        if last_bpath is None:
                            logging.error(f"Good frame without a preceding bad frame: {path}")
                            error = True
                        else:
                            groups_by_ig[ig].append(PathPair(last_bpath, path))
                            last_bpath = None

                else:
                    logging.warning(f"Skipping file with unexpected name format: {path}")


            if last_bpath is not None:
                logging.error(f"Bad frame without a corresponding good frame: {last_bpath}")
                error = True

            if not groups_by_ig:
                logging.warning(f"No valid path pairs found in directory: {sub_dir}")
                continue

            ppair_groups.extend(groups_by_ig.values())

        if error:
            raise ValueError("Errors detected while loading path pairs. Check the log for details.")

        return ppair_groups



