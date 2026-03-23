# USAGE:
# python -m manim render -qh src/main.py <scene class name>
# Example:
# python -m manim render -qh src/main.py HistogramEqualization
# -qh = high quality (1080p)
# -ql = low quality (480p)

import sys
import os
import tempfile
import numpy as np
from PIL import Image
from manim import *

sys.path.insert(0, os.path.dirname(__file__))
from filters.histogram_equalizer import equalize_histogram

image_paths = {
    "histogram_equalization": os.path.join(os.path.dirname(__file__), "..", "assets", "dog_quality_bad.png"),
}

class HistogramEqualization(Scene):
    def construct(self):
        image_path = image_paths["histogram_equalization"]

        original_arr, equalized_arr = equalize_histogram(image_path)

        # Save arrays as temporary images for Manim to load
        tmp_dir = tempfile.mkdtemp()
        orig_path = os.path.join(tmp_dir, "original.png")
        eq_path = os.path.join(tmp_dir, "equalized.png")
        Image.fromarray(original_arr).save(orig_path)
        Image.fromarray(equalized_arr).save(eq_path)

        # --- Title ---
        title = Text("Histogram Equalization", font_size=36).to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1)

        # --- Formula ---
        formula = MathTex(
            r"h(v) = \text{round}\!\left(\frac{\text{cdf}(v) - \text{cdf}_{\min}}{M \times N - \text{cdf}_{\min}} \cdot (L-1)\right)",
            font_size=30,
        ).to_edge(DOWN, buff=0.4)

        # --- Images ---
        img_height = 5.0
        original_img = ImageMobject(orig_path).set_height(img_height)
        equalized_img = ImageMobject(eq_path).set_height(img_height)

        # Position images side by side (will animate from center)
        original_img.move_to(ORIGIN)
        equalized_img.move_to(ORIGIN)

        # --- Labels ---
        orig_label = Text("Original", font_size=24, color=BLUE)
        eq_label = Text("Equalized", font_size=24, color=GREEN)

        # Show original image first
        self.play(FadeIn(original_img), run_time=1)
        self.wait(0.5)

        # Animate original sliding left, equalized fading in on the right
        original_img.generate_target()
        original_img.target.shift(LEFT * 3.2)

        equalized_img.shift(RIGHT * 3.2)
        equalized_img.set_opacity(0)

        # Build blended transition frames
        n_steps = 30
        blend_imgs = []
        blend_paths = []
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            blended = (
                (1 - alpha) * original_arr.astype(np.float64)
                + alpha * equalized_arr.astype(np.float64)
            ).astype(np.uint8)
            p = os.path.join(tmp_dir, f"blend_{i}.png")
            Image.fromarray(blended).save(p)
            blend_paths.append(p)

        # Smooth pixel-level transition in place
        for p in blend_paths:
            new_img = ImageMobject(p).set_height(img_height).move_to(original_img.get_center())
            self.remove(original_img)
            original_img = new_img
            self.add(original_img)
            self.wait(1 / 30)

        self.wait(0.3)

        # Slide equalized image to the right, show original on the left simultaneously
        original_final = ImageMobject(orig_path).set_height(img_height).move_to(LEFT * 3.8)
        original_final.set_opacity(0)

        self.add(original_final)
        self.play(
            original_img.animate.move_to(RIGHT * 3.8),
            original_final.animate.set_opacity(1),
            run_time=1,
        )

        # Labels under images
        orig_label.next_to(original_final, DOWN, buff=0.2)
        eq_label.next_to(original_img, DOWN, buff=0.2)
        self.play(Write(orig_label), Write(eq_label), run_time=0.8)

        # Show formula
        self.play(Write(formula), run_time=1.5)
        self.wait(2)

        # Cleanup temp files
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)