import argparse
from collections.abc import Iterable
import re
from pathlib import Path

def dump_frame_pairs_from_raw_dir(ref_dir: Path, frame_pairs_path: Path):
    
    # regex for frame files: frame_<frame_number>g.png or frame_<frame_number>.png
    namepat = re.compile(r'frame_(\d+)(g)?\.png')

    vdirs = sorted(d for d in ref_dir.iterdir() if d.is_dir())

    # 各動画ディレクトリを走査
    fgroups: list[list[int]] = [] # list(video)[list(pair)[tuple[igood, ibad1, ibad2, ...]]]
    haserror = False
    for vdir in vdirs:

        print(f"Checking good/bad frame pairs on {vdir}")

        lasti = -2
        for fpath in sorted(vdir.iterdir()):
            if not (m := namepat.match(fpath.name)):
                continue

            i = int(m.group(1))
            isgood = m.group(2) == 'g'

            if lasti + 1 != i:
                fgroups.append([-1])
            lasti = i

            if isgood:
                if fgroups[-1][0] != -1:
                    print(f"Dataset Error: Multiple good frames on frame {i}")
                    haserror = True
                fgroups[-1][0] = i
            else:
                fgroups[-1].append(i)

        for ig, *ibs in fgroups:
            if ig == -1:
                print(f"Dataset Error: No good frame for bad frame(s) {', '.join(map(str, ibs))}")
            if not len(ibs):
                print(f"Dataset Error: No bad frames for good frame {ig}")

        if haserror:
            raise RuntimeError("Dataset has error(s).")
    
    pairs = sorted((ig, ib) for ig, *ibs in fgroups for ib in ibs)
    dump_frame_pairs(pairs, frame_pairs_path)
        

def dump_frame_pairs_from_proessed_dir(ref_dir: Path, frame_pairs_path: Path):
    imgpattern = re.compile(r'ig(\d+)_ib(\d+)')
    pairs: set[tuple[int, int]] = set() # set of (good, bad)

    for img in ref_dir.glob('*.png'):
        if m := imgpattern.match(img.name):
            ig = int(m[1]) # Good frame index
            ib = int(m[2]) # Bad frame index
            pairs.add((ig, ib))
        else:
            print('Warning: unknown file: ' + img.name)

    dump_frame_pairs(pairs, frame_pairs_path)


def dump_frame_pairs(pairs: Iterable[tuple[int, int]], frame_pairs_path: Path):
    with frame_pairs_path.open('w') as fw:
        print("#good\tbad", file=fw)
        for ig, ib in sorted(list(pairs)):
            print(f"{ig}\t{ib}", file=fw)


def load_frame_pairs(dumppath: Path):
    
    pairs: list[tuple[int, int]] = [] # pairs of frame index
    with dumppath.open() as f:
        for _line in f:
            line = _line.strip()
            if line[0] == '#':
                continue
            _ig, _ib = line.strip().split('\t')
            ig, ib = int(_ig), int(_ib)
            pairs.append((ig, ib))
    return pairs


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('-rd', '--raw-dir', type=str)
    argp.add_argument('-pd', '--processed-dir', type=str)
    argp.add_argument('-o', '--output', type=str, required=True)
    args = argp.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir is not None else None
    processed_dir = Path(args.processed_dir) if args.processed_dir is not None else None
    output = Path(args.output)

    if not raw_dir and not processed_dir:
        raise RuntimeError("Both of --raw-dir and --processed-dir are not specified.")
    
    if raw_dir:
        dump_frame_pairs_from_raw_dir(raw_dir, output)
    
    if processed_dir:
        dump_frame_pairs_from_proessed_dir(processed_dir, output)


if __name__ == '__main__':
    main()
