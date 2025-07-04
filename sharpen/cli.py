import argparse
import getpass

from .push_image_dataset_to_hub import push_images

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
