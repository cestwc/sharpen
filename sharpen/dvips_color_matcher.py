#!/usr/bin/env python3
"""
LaTeX Color Matcher
===================

A tool to find the best approximation of a target RGB/Hex color using 
standard LaTeX colors (dvipsnames) and the xcolor '!' mixing syntax.

Information Source:
    Color definitions based on: https://en.wikibooks.org/wiki/LaTeX/Colors
    Default Data Source: https://github.com/cestwc/sharpen/releases/download/v1.0.0/dvipsnames.csv
"""

import math
import heapq
import pandas as pd

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
DEFAULT_CSV_URL = "https://github.com/cestwc/sharpen/releases/download/v1.0.0/dvipsnames.csv"

# -------------------------------------------------------------------------
# MATH ENGINE
# -------------------------------------------------------------------------

def parse_hex(h):
	h = h.lstrip('#')
	return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))

def cmyk_mix(c1, c2, p):
	""" xcolor mixing formula: p% of c1 + (100-p)% of c2 """
	ratio = p / 100.0
	return tuple(ratio * a + (1.0 - ratio) * b for a, b in zip(c1, c2))

def cmyk_to_rgb_naive(cmyk):
	""" Generic PDF viewer simulation (R = (1-C)(1-K)) """
	c, m, y, k = cmyk
	r = (1.0 - c) * (1.0 - k)
	g = (1.0 - m) * (1.0 - k)
	b = (1.0 - y) * (1.0 - k)
	return (r, g, b)

def rgb_to_cmyk_naive(r, g, b):
	k = 1 - max(r, g, b)
	if k == 1: return (0, 0, 0, 1)
	return ((1-r-k)/(1-k), (1-g-k)/(1-k), (1-b-k)/(1-k), k)

def rgb_to_lab(rgb):
	""" sRGB -> XYZ -> CIELAB (D65) """
	r, g, b = rgb
	func = lambda c: c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
	r, g, b = map(func, (r, g, b))
	X = r*0.4124 + g*0.3576 + b*0.1805
	Y = r*0.2126 + g*0.7152 + b*0.0722
	Z = r*0.0193 + g*0.1192 + b*0.9505
	xn, yn, zn = 0.95047, 1.0, 1.08883
	func2 = lambda t: t**(1/3) if t > 0.008856 else 7.787*t + 16/116
	return (116*func2(Y/yn)-16, 500*(func2(X/xn)-func2(Y/yn)), 200*(func2(Y/yn)-func2(Z/zn)))

# --- Metrics ---

def distance_rgb_euclidean(rgb1, rgb2):
	""" Simple Euclidean distance on 0-255 scale """
	r1, g1, b1 = [x * 255.0 for x in rgb1]
	r2, g2, b2 = [x * 255.0 for x in rgb2]
	return math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)

def delta_e_ciede2000(lab1, lab2):
	""" CIEDE2000 Standard for Perceptual Difference """
	L1, a1, b1 = lab1; L2, a2, b2 = lab2
	avg_L = (L1 + L2) / 2.0
	C1 = math.sqrt(a1**2 + b1**2); C2 = math.sqrt(a2**2 + b2**2); avg_C = (C1 + C2) / 2.0
	G = 0.5 * (1 - math.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
	a1_p = (1 + G) * a1; a2_p = (1 + G) * a2
	C1_p = math.sqrt(a1_p**2 + b1**2); C2_p = math.sqrt(a2_p**2 + b2**2); avg_C_p = (C1_p + C2_p) / 2.0
	if C1_p == 0:
		h1_p = 0
	else:
		h1_p = math.degrees(math.atan2(b1, a1_p)) % 360
	if C2_p == 0:
		h2_p = 0
	else:
		h2_p = math.degrees(math.atan2(b2, a2_p)) % 360
	if abs(h1_p - h2_p) <= 180:
		delta_h_p = h2_p - h1_p
	elif h2_p <= h1_p:
		delta_h_p = h2_p - h1_p + 360
	else:
		delta_h_p = h2_p - h1_p - 360
	delta_L_p = L2 - L1; delta_C_p = C2_p - C1_p
	delta_H_p = 2 * math.sqrt(C1_p * C2_p) * math.sin(math.radians(delta_h_p) / 2.0)
	avg_L_p = (L1 + L2) / 2.0; 
	if abs(h1_p - h2_p) <= 180:
		avg_h_p = (h1_p + h2_p) / 2.0
	elif abs(h1_p - h2_p) > 180 and (h1_p + h2_p) < 360:
		avg_h_p = (h1_p + h2_p + 360) / 2.0
	else:
		avg_h_p = (h1_p + h2_p - 360) / 2.0
	T = 1 - 0.17 * math.cos(math.radians(avg_h_p - 30)) + 0.24 * math.cos(math.radians(2 * avg_h_p)) + 0.32 * math.cos(math.radians(3 * avg_h_p + 6)) - 0.20 * math.cos(math.radians(4 * avg_h_p - 63))
	S_L = 1 + (0.015 * (avg_L_p - 50)**2) / math.sqrt(20 + (avg_L_p - 50)**2)
	S_C = 1 + 0.045 * avg_C_p; S_H = 1 + 0.015 * avg_C_p * T
	delta_theta = 30 * math.exp(-((avg_h_p - 275) / 25)**2)
	R_C = 2 * math.sqrt(avg_C_p**7 / (avg_C_p**7 + 25**7))
	R_T = -math.sin(math.radians(2 * delta_theta)) * R_C
	return math.sqrt((delta_L_p / S_L)**2 + (delta_C_p / S_C)**2 + (delta_H_p / S_H)**2 + R_T * (delta_C_p / S_C) * (delta_H_p / S_H))

# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------
def load_data():
	""" Downloads color definitions and injects the 19 standard LaTeX base colors. """
	print(f"Downloading data from: {DEFAULT_CSV_URL}")
	colors = {}
	
	# 1. Load the fancy names (dvipsnames) from CSV
	try:
		df = pd.read_csv(DEFAULT_CSV_URL)
		df.columns = df.columns.str.strip().str.lower()
		
		for _, row in df.iterrows():
			if 'name' not in row or 'hex' not in row: continue
			name = str(row['name']).strip()
			hex_val = str(row['hex']).strip().lstrip('#')
			if len(hex_val) != 6: continue
			
			try:
				r = int(hex_val[0:2], 16) / 255.0
				g = int(hex_val[2:4], 16) / 255.0
				b = int(hex_val[4:6], 16) / 255.0
				colors[name] = rgb_to_cmyk_naive(r, g, b)
			except: continue
	except Exception as e:
		print(f"Warning: Could not load CSV ({e}). Using base colors only.")
	
	# 2. Inject the 19 Standard LaTeX Base Colors (RGB definitions)
	# These match the standard xcolor definitions shown in your image.
	standard_bases_rgb = {
		"red":       (1.0, 0.0, 0.0),
		"green":     (0.0, 1.0, 0.0),
		"blue":      (0.0, 0.0, 1.0),
		"cyan":      (0.0, 1.0, 1.0),
		"magenta":   (1.0, 0.0, 1.0),
		"yellow":    (1.0, 1.0, 0.0),
		"black":     (0.0, 0.0, 0.0),
		"white":     (1.0, 1.0, 1.0),
		"gray":      (0.5, 0.5, 0.5),
		"darkgray":  (0.25, 0.25, 0.25),
		"lightgray": (0.75, 0.75, 0.75),
		"brown":     (0.75, 0.5, 0.25),
		"lime":      (0.75, 1.0, 0.0),
		"olive":     (0.5, 0.5, 0.0),
		"orange":    (1.0, 0.5, 0.0),
		"pink":      (1.0, 0.75, 0.75),
		"purple":    (0.75, 0.0, 0.25),
		"teal":      (0.0, 0.5, 0.5),
		"violet":    (0.5, 0.0, 0.5)
	}
	
	# Convert these RGBs to your script's CMYK format and add them
	for name, (r, g, b) in standard_bases_rgb.items():
		# Only add if not already present (standard bases usually take precedence)
		# Note: We overwrite here to ensure 'green' is PURE green, not dvips Green.
		colors[name] = rgb_to_cmyk_naive(r, g, b)
	
	print(f" -> Loaded {len(colors)} valid colors (including {len(standard_bases_rgb)} standard bases).")
	return colors
	
# -------------------------------------------------------------------------
# OPTIMIZER
# -------------------------------------------------------------------------
def solve(target_hex, max_bangs, metric, beam_width, step_size):
	base_colors = load_data()
	
	t_rgb = parse_hex(target_hex)
	t_lab = rgb_to_lab(t_rgb) if metric == 'lab' else None
	
	def get_gap(cand_rgb):
		if metric == 'lab':
			return delta_e_ciede2000(rgb_to_lab(cand_rgb), t_lab)
		else:
			return distance_rgb_euclidean(cand_rgb, t_rgb)
	
	# Depth 0
	current_gen = []
	for name, vec in base_colors.items():
		c_rgb = cmyk_to_rgb_naive(vec)
		gap = get_gap(c_rgb)
		current_gen.append((gap, vec, name, c_rgb))
	
	best_results = {0: min(current_gen, key=lambda x: x[0])}
	
	# Depth 1..N
	for k in range(1, max_bangs + 1):
		next_gen = []
		candidates = heapq.nsmallest(beam_width, current_gen, key=lambda x: x[0])
		
		for (c_gap, c_vec, c_expr, c_rgb_val) in candidates:
			# Mix
			for b_name, b_vec in base_colors.items():
				if c_expr.endswith(b_name): continue
				for p in range(step_size, 100, step_size):
					n_vec = cmyk_mix(c_vec, b_vec, p)
					n_expr = f"{c_expr}!{p}!{b_name}"
					n_rgb = cmyk_to_rgb_naive(n_vec)
					n_gap = get_gap(n_rgb)
					next_gen.append((n_gap, n_vec, n_expr, n_rgb))
			
			# Implicit White (Depth 1 only)
			if k == 1 and "!" not in c_expr:
				for p in range(step_size, 100, step_size):
					n_vec = cmyk_mix(c_vec, base_colors["White"], p)
					n_expr = f"{c_expr}!{p}!White"
					n_rgb = cmyk_to_rgb_naive(n_vec)
					n_gap = get_gap(n_rgb)
					next_gen.append((n_gap, n_vec, n_expr, n_rgb))
	
		if next_gen:
			best_results[k] = min(next_gen, key=lambda x: x[0])
			current_gen = next_gen
			
	return t_rgb, best_results

# -------------------------------------------------------------------------
# LATEX GENERATOR
# -------------------------------------------------------------------------
def generate_latex(target_hex, results, metric_name):
	hex_clean = target_hex.lstrip('#').upper()
	rows = ""
	for k in sorted(results.keys()):
		gap, _, expr, rgb_val = results[k]
		rgb_str = f"({rgb_val[0]*255:.1f}, {rgb_val[1]*255:.1f}, {rgb_val[2]*255:.1f})"
		rows += rf"""
	{k} & {gap:.2f} & \small\texttt{{{rgb_str}}} &
	\colorbox{{Target}}{{\rule{{0pt}}{{1.5em}}\rule{{1.5em}}{{0pt}}}}%
	\colorbox{{{expr}}}{{\rule{{0pt}}{{1.5em}}\rule{{1.5em}}{{0pt}}}} & 
	\small\texttt{{{expr}}} \\"""
	
	return rf"""\documentclass{{article}}
\usepackage[dvipsnames]{{xcolor}}
\usepackage[margin=1in]{{geometry}}
\definecolor{{Target}}{{HTML}}{{{hex_clean}}}
\begin{{document}}
\section*{{Color Matcher: \#{hex_clean}}}
\textbf{{Metric:}} {metric_name}
\renewcommand{{\arraystretch}}{{2}}
\begin{{tabular}}{{c l l l l}}
    \textbf{{k}} & \textbf{{Diff}} & \textbf{{Simulated RGB}} & \textbf{{Vis}} & \textbf{{Code}} \\ \hline
    {rows}
\end{{tabular}}
\end{{document}}"""

# -------------------------------------------------------------------------
# MAIN CLI
# -------------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser(description="Find LaTeX xcolor approximations for a target Hex color.")
	parser.add_argument("hex", help="Target Hex Code (e.g. #3450a0)")
	parser.add_argument("-n", "--bangs", type=int, default=2, help="Max depth of mixing (default: 2)")
	parser.add_argument("-m", "--metric", choices=['rgb', 'lab'], default='rgb', 
						help="Distance metric: 'rgb' (Euclidean) or 'lab' (CIEDE2000)")
	parser.add_argument("--csv", default=DEFAULT_CSV_URL, help="URL to dvipsnames.csv")
	parser.add_argument("--beam", type=int, default=1000, help="Beam search width (default: 1000)")
	parser.add_argument("--step", type=int, default=5, help="Mixing step size (default: 5)")
	parser.add_argument("--tex", action="store_true", help="Generate LaTeX report file")
	parser.add_argument("--output", default="color_match.tex", help="Output filename for LaTeX report")
	
	args = parser.parse_args()
	   
	print(f"Target: {args.hex}")
	print(f"Config: n={args.bangs}, beam={args.beam}, step={args.step}")
	print(f"Metric: {args.metric.upper()} distance")
	
	t_rgb, res = solve(args.hex, args.bangs, args.metric, args.beam, args.step)
	
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

if __name__ == "__main__":
	import argparse
	main()
