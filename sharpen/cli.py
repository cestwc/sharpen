import argparse
import getpass

from .push_image_dataset_to_hub import push_images
from .dvips_color_matcher import solve as dvips_solve

def main():
	parser = argparse.ArgumentParser(prog="sharpen", description="Sharpen CLI tool")
	subparsers = parser.add_subparsers(dest="command", required=True)
	
	# Subcommand: push
	push = subparsers.add_parser("push-images", help="Push image dataset to Hugging Face Hub")
	push.add_argument("--images", required=True, help="Folder, tensor, or array input")
	push.add_argument("--labels", required=True, help="Label list or .parquet path")
	push.add_argument("--repo", required=True, help="HF dataset repo name")
	push.add_argument("--token", required=False, help="HF token")
	push.add_argument("--config-name", required=True, help="Dataset config/version")
	push.add_argument("--class-names", default='', help="Class names: list, 'cifar10', or HF dataset")
	push.add_argument("--private", action="store_true", help="Make dataset private")
	push.add_argument("--image-sort-mode", default="natural", help="Sort mode: natural/plain/mtime/none")

	# Subcommand: push
	dvips = subparsers.add_parser("dvips", help="Approximation of a target Hex color using dvipsnames")
	dvips.add_argument("hex", help="Target Hex Code (e.g. #3450a0)")
	dvips.add_argument("-n", "--bangs", type=int, default=2, help="Max depth of mixing (default: 2)")
	dvips.add_argument("-m", "--metric", choices=['rgb', 'lab'], default='rgb', 
						help="Distance metric: 'rgb' (Euclidean) or 'lab' (CIEDE2000)")
	dvips.add_argument("--beam", type=int, default=1000, help="Beam search width (default: 1000)")
	dvips.add_argument("--step", type=int, default=5, help="Mixing step size (default: 5)")
	dvips.add_argument("--tex", action="store_true", help="Generate LaTeX report file")
	dvips.add_argument("--output", default="color_match.tex", help="Output filename for LaTeX report")
	
	args = parser.parse_args()
	
	if args.command == "push-images":
		token = args.token or getpass.getpass("Enter Hugging Face token: ")
		
		# Load labels from .txt if applicable
		labels = args.labels
		if isinstance(labels, str) and labels.endswith(".txt"):
			with open(labels) as f:
				labels = [int(x.strip()) for x in f]
		
		push_images(
			images=args.images,
			repo=args.repo,
			token=token,
			config_name=args.config_name,
			labels=labels,
			class_names=args.class_names,
			private=args.private,
			image_sort_mode=args.image_sort_mode
		)
	
	
	
	if args.command == "dvips":
		
		print(f"Target: {args.hex}")
		print(f"Config: n={args.bangs}, beam={args.beam}, step={args.step}")
		print(f"Metric: {args.metric.upper()} distance")
		
		t_rgb, res = dvips_solve(args.hex, args.bangs, args.metric, args.beam, args.step)
		
		print("-" * 85)
		print(f"{'k':<3} | {'Diff':<8} | {'Simulated RGB':<22} | {'Expression'}")
		print("-" * 85)
		for k in sorted(res.keys()):
			gap, _, expr, rgb_val = res[k]
			rgb_str = f"({rgb_val[0]*255:.1f}, {rgb_val[1]*255:.1f}, {rgb_val[2]*255:.1f})"
			print(f"{k:<3} | {gap:<8.2f} | {rgb_str:<22} | {expr}")
		print("-" * 85)
		
		if args.tex:
			with open(args.output, "w") as f:
				f.write(generate_latex(args.hex, res, args.metric.upper()))
			print(f"LaTeX report saved to: {args.output}")
