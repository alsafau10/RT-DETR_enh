"""by lyuwenyu"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def main(args) -> None:
    """main"""
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    # Load YAML config
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )

    # ── Attach our enhancement settings so solver can read them ────────
    cfg.enhance_method = args.enhance_method
    cfg.he_threshold  = args.he_threshold
    cfg.clahe_clip    = args.clahe_clip
    cfg.clahe_grid    = tuple(args.clahe_grid)
    base = Path(cfg.output_dir)
    tagged = f"{base.name}_{cfg.enhance_method}{int(cfg.he_threshold * 100)}"
    cfg.output_dir = str(base.parent / tagged)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    '-c', type=str, help='path to config file')
    parser.add_argument('--resume',    '-r', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--tuning',    '-t', type=str, default=None, help='tuning from checkpoint')
    parser.add_argument('--test-only',        action='store_true', default=False, help='only test')
    parser.add_argument('--amp',              action='store_true', default=False, help='use AMP')
    parser.add_argument('--seed',      type=int, default=None, help='random seed')

    # ── NEW ENHANCEMENT FLAGS ─────────────────────────────────────────────
    parser.add_argument(
        '--enhance-method',
        type=str,
        default='none',
        choices=['none', 'he', 'linear', 'clahe'],
        help='Which enhancement to run per-epoch'
    )
    parser.add_argument(
        '--he-threshold',
        type=float,
        default=1.0,
        help='Probability threshold [0–1] for applying enhancement each epoch'
    )
    parser.add_argument(
        '--clahe-clip',
        type=float,
        default=2.0,
        help='clipLimit for CLAHE (only used if --enhance-method clahe)'
    )
    parser.add_argument(
        '--clahe-grid',
        type=int,
        nargs=2,
        default=[8, 8],
        help='tileGridSize (h w) for CLAHE'
    )

    args = parser.parse_args()
    main(args)
