import argparse
import sys
import yaml

def load_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True, help="YAML config files")

    parser.add_argument('--output_dir')
    parser.add_argument('--resume_files', nargs='+')
    parser.add_argument('--resume_optimizer', default=None)

    parser.add_argument('--test', default=False, action='store_true')

    # distributed computing
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="local rank for distributed training on gpus",
    )
    parser.add_argument(
        "--node_rank", type=int, default=0,
        help="Id of the node",
    )
    parser.add_argument(
        "--world_size", type=int, default=1,
        help="Number of GPUs across all nodes",
    )

    return parser

def parse_with_config(parser):
    args = parser.parse_args()

    if args.config is not None:
        config_args = yaml.safe_load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config

    return args