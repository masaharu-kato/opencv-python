import argparse
import logging
import re
from pathlib import Path

from semver import process

from path_groups import PathGroup, PathGroups


def make_path_group_from_continuous_frames_dir(frames_dir: Path) -> PathGroups:
    # regex for frame files: frame_<frame_number>g.png or frame_<frame_number>.png
    namepat = re.compile(r'.*(\d+)(g)?\.png')

    path_groups = PathGroups()
    last_gpath: Path | None = None
    last_bpaths: list[Path] = []
    count = 0

    haserror = False
    def process_last_group():
        nonlocal last_gpath, last_bpaths, haserror, count
        if last_gpath is not None and last_bpaths:
            path_groups.append(PathGroup.load(last_gpath, last_bpaths))
            count += 1
        elif last_gpath is not None:
            logging.error(f"{last_gpath}: No bad frames found.")
            haserror = True
        elif last_bpaths:
            logging.error(f"{', '.join(map(str, last_bpaths))}: No good frame found.")
            haserror = True
        last_gpath = None
        last_bpaths = []


    logging.info(f"Checking good/bad frame pairs on {frames_dir}")

    for sub_dir in frames_dir.iterdir():
        if not sub_dir.is_dir():
            continue

        logging.info(f"Processing subdirectory: {sub_dir}")

        lasti = -2
        count = 0

        for fpath in sorted(sub_dir.glob("*.png")):
            if not (m := namepat.match(fpath.name)):
                logging.warning(f"{fpath}: Invalid frame name format, skipping.")
                continue
            
            i = int(m.group(1))
            isgood = m.group(2) == 'g'

            if lasti + 1 != i:
                process_last_group()
                
            lasti = i

            if isgood:
                if last_gpath is not None:
                    logging.error(f"{fpath}: Good frame alreay exists: {last_gpath}")
                    haserror = True
                last_gpath = fpath
            else:
                last_bpaths.append(fpath)

        process_last_group()

        if haserror:
            raise RuntimeError("Dataset has error(s).")
        
        if count == 0:
            logging.warning(f"No frames found in {sub_dir}")
    
    return path_groups


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Make PathGroups from continuous frames directory.")
    parser.add_argument("frames_dir", type=Path, help="Directory containing continuous frame images.")
    
    args = parser.parse_args()
    frames_dir = Path(args.frames_dir)

    path_groups = make_path_group_from_continuous_frames_dir(frames_dir)
    out_path = frames_dir / "path_groups.tsv"
    path_groups.dump_to_file(out_path)

    logging.info(f"Saved {len(path_groups)} PathGroups to {out_path}")


if __name__ == "__main__":
    main()
