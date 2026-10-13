"""
synthesize.py — Synthetic Receipt Data & Noise Generator.

Programmatically generates realistic receipt images with:
  1. Multilingual item tables (Russian, Uzbek Cyrillic, Uzbek Latin, English).
  2. Ground-truth metadata (SKUs, names, units, quantities, unit prices, totals).
  3. Controlled physical degradations for stress-testing and benchmarking:
     - Gaussian / Motion blur
     - Lighting gradients & shadow cast
     - Handwritten blue ink checkmarks / pen scratches
     - Gaussian sensor noise
     - Perspective tilt / rotation

Usage:
    # Generate 5 clean receipts with ground truth:
    python tools/synthesize.py --count 5 --output data/synthetic/

    # Generate noisy receipts with shadow and handwritten blue checkmarks:
    python tools/synthesize.py --count 5 --output data/synthetic/ --noise shadow,pen_marks,blur
"""

import argparse
import json
import math
from pathlib import Path
import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SAMPLE_PRODUCTS = [
    # (name, default_unit, min_price, max_price, default_sku)
    ("Парф. вода Incandessence 50ml", "шт", 85000, 110000, "F2664000"),
    ("Туал. вода Perceive Dew 50ml", "шт", 95000, 120000, "F5761400"),
    ("Гель-маска для лица Avon 8мл", "шт", 7000, 15000, "F1539984"),
    ("Olma sharbati 1l (O'zbekiston)", "л", 12000, 18000, "UZ991021"),
    ("Choy Ahmad English Breakfast 100g", "уп", 24000, 32000, "AHM0021"),
    ("Sut Musaffo 3.2% 1 litr", "л", 11500, 14000, "MSF3201"),
    ("Скраб для тела увлажняющий 200мл", "шт", 35000, 48000, "SKR2001"),
    ("Печенье OREO классическое 95г", "уп", 8500, 12000, "ORO9500"),
    ("Qand / Shakar oq tozalangan", "кг", 9000, 13000, "SHK1001"),
    ("Кофе Nescafe Gold 190г", "банка", 78000, 95000, "NSF1900"),
    ("Makaron Gallina Blanca 400g", "уп", 9500, 13500, "GLB4001"),
    ("Mineral suv Chortoq 0.5l gazli", "бутылка", 4500, 6500, "CRT0500"),
]

STORE_NAMES = [
    "OOO 'KORZINKA TRADE' — MARKET",
    "SUPERMARKET 'MAKRO' TASHKENT",
    "DO'KON 'BARAKA BOZOR'",
    "IP 'FAMILY MARKET' CHILANZAR",
]


class SyntheticReceiptGenerator:
    """
    Generates synthetic receipts with ground truth annotations and optional noise.
    """

    def __init__(self, font_size: int = 22):
        self.font_size = font_size

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """Attempts to load a standard readable font, falls back to default."""
        try:
            # Common macOS / Linux TTF fonts supporting Cyrillic + Latin
            for font_name in [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "Arial.ttf",
            ]:
                if Path(font_name).exists():
                    return ImageFont.truetype(font_name, size)
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def generate(
        self,
        num_items: int = 4,
        noise_types: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Generates a synthetic receipt.

        Returns:
            (bgr_image, ground_truth_dict)
        """
        noise_types = noise_types or []
        items_pool = random.sample(SAMPLE_PRODUCTS, min(num_items, len(SAMPLE_PRODUCTS)))

        products = []
        total_sum = 0.0

        for idx, (name, unit, min_p, max_p, sku) in enumerate(items_pool, 1):
            qty = random.choice([1, 2, 3, 5])
            price = random.randint(min_p // 500, max_p // 500) * 500
            line_sum = qty * price
            total_sum += line_sum

            products.append({
                "line_no": idx,
                "sku": sku,
                "name": name,
                "quantity": qty,
                "unit": unit,
                "price": price,
                "total": line_sum,
            })

        store_name = random.choice(STORE_NAMES)

        # ── Render with Pillow ──────────────────────────────────────────
        width = 800
        row_height = 42
        header_height = 200
        table_height = (len(products) + 1) * row_height
        footer_height = 160
        total_height = header_height + table_height + footer_height

        img = Image.new("RGB", (width, total_height), color=(252, 252, 250))
        draw = ImageDraw.Draw(img)

        font_large = self._get_font(24)
        font_regular = self._get_font(18)
        font_small = self._get_font(15)

        # Header
        draw.text((width // 2, 30), store_name, fill=(20, 20, 20), font=font_large, anchor="mt")
        draw.text((width // 2, 65), "ИНН: 305882194  |  Фискальный чек № 004812", fill=(60, 60, 60), font=font_small, anchor="mt")
        draw.text((width // 2, 90), "г. Ташкент, ул. Амира Темура, 45", fill=(80, 80, 80), font=font_small, anchor="mt")
        draw.text((width // 2, 115), "Дата: 12.10.2026 14:32:10", fill=(80, 80, 80), font=font_small, anchor="mt")

        draw.line([(40, 150), (width - 40, 150)], fill=(120, 120, 120), width=2)

        # Table Header
        col_x = [50, 100, 220, 480, 550, 640, 750]  # boundaries/anchors
        y = 165
        headers = ["№", "Код", "Наименование товара", "Кол", "Ед.", "Цена", "Сумма"]
        draw.text((50, y), "№", fill=(30, 30, 30), font=font_regular)
        draw.text((90, y), "Код", fill=(30, 30, 30), font=font_regular)
        draw.text((190, y), "Наименование", fill=(30, 30, 30), font=font_regular)
        draw.text((460, y), "Кол-во", fill=(30, 30, 30), font=font_regular)
        draw.text((530, y), "Ед.", fill=(30, 30, 30), font=font_regular)
        draw.text((600, y), "Цена", fill=(30, 30, 30), font=font_regular)
        draw.text((700, y), "Сумма", fill=(30, 30, 30), font=font_regular)

        draw.line([(40, y + 28), (width - 40, y + 28)], fill=(150, 150, 150), width=1)
        y += 35

        # Table Rows
        for p in products:
            draw.text((50, y), str(p["line_no"]), fill=(20, 20, 20), font=font_regular)
            draw.text((90, y), p["sku"], fill=(20, 20, 20), font=font_regular)
            draw.text((190, y), p["name"], fill=(20, 20, 20), font=font_regular)
            draw.text((470, y), str(p["quantity"]), fill=(20, 20, 20), font=font_regular)
            draw.text((530, y), p["unit"], fill=(20, 20, 20), font=font_regular)
            draw.text((590, y), f"{p['price']:,}".replace(",", " "), fill=(20, 20, 20), font=font_regular)
            draw.text((690, y), f"{p['total']:,}".replace(",", " "), fill=(20, 20, 20), font=font_regular)
            y += row_height

        draw.line([(40, y), (width - 40, y)], fill=(120, 120, 120), width=2)
        y += 20

        # Footer
        draw.text((450, y), "ИТОГО К ОПЛАТЕ:", fill=(10, 10, 10), font=font_large)
        draw.text((680, y), f"{int(total_sum):,} UZS".replace(",", " "), fill=(10, 10, 10), font=font_large)
        y += 45
        draw.text((width // 2, y), "*** СПАСИБО ЗА ПОКУПКУ! / XARIDINGIZ UCHUN RAXMAT! ***", fill=(100, 100, 100), font=font_small, anchor="mt")

        # Convert to OpenCV BGR
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # ── Apply Injected Noise / Degradations ────────────────────────
        if "shadow" in noise_types:
            np_img = self._apply_shadow_gradient(np_img)
        if "pen_marks" in noise_types:
            np_img = self._apply_pen_marks(np_img, num_marks=len(products))
        if "blur" in noise_types:
            np_img = cv2.GaussianBlur(np_img, (5, 5), sigmaX=1.5)
        if "noise" in noise_types:
            np_img = self._apply_gaussian_noise(np_img)
        if "rotation" in noise_types:
            np_img = self._apply_rotation(np_img, angle=random.uniform(-2.5, 2.5))

        ground_truth = {
            "store_name": store_name,
            "total_items": len(products),
            "total_sum": total_sum,
            "products": products,
            "noise_applied": noise_types,
        }

        return np_img, ground_truth

    # ------------------------------------------------------------------
    # Synthetic Noise Injection Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_shadow_gradient(img: np.ndarray) -> np.ndarray:
        """Simulates uneven lighting / shadows across the receipt."""
        h, w = img.shape[:2]
        x_grad = np.linspace(0.65, 1.0, w)
        y_grad = np.linspace(0.85, 1.0, h)
        xx, yy = np.meshgrid(x_grad, y_grad)
        gradient = (xx * yy)[:, :, np.newaxis]
        shadowed = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        return shadowed

    @staticmethod
    def _apply_pen_marks(img: np.ndarray, num_marks: int = 3) -> np.ndarray:
        """
        Draws handwritten blue checkmarks across receipt lines in the
        HSV range (100–130) matching physical ballpoint pen ink.
        """
        out = img.copy()
        h, w = out.shape[:2]
        pen_bgr = (180, 50, 20)  # Classic blue ballpoint pen (BGR)

        for _ in range(num_marks):
            cx = random.randint(30, 120)
            cy = random.randint(180, h - 100)
            # Draw a checkmark: short diagonal down, long diagonal up
            pts = np.array([
                [cx - 15, cy - 5],
                [cx, cy + 12],
                [cx + 35, cy - 25]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            thickness = random.randint(2, 4)
            cv2.polylines(out, [pts], isClosed=False, color=pen_bgr, thickness=thickness, lineType=cv2.LINE_AA)

        return out

    @staticmethod
    def _apply_gaussian_noise(img: np.ndarray, std: float = 12.0) -> np.ndarray:
        """Adds sensor shot noise."""
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    @staticmethod
    def _apply_rotation(img: np.ndarray, angle: float = 2.0) -> np.ndarray:
        """Rotates image slightly with white border padding."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return rotated


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic receipt images for testing and benchmarking.")
    parser.add_argument("--count", type=int, default=3, help="Number of synthetic receipts to generate (default: 3).")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory.")
    parser.add_argument("--items", type=int, default=4, help="Number of items per receipt (default: 4).")
    parser.add_argument("--noise", type=str, default="", help="Comma-separated noise list: blur,shadow,pen_marks,noise,rotation.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_types = [n.strip() for n in args.noise.split(",") if n.strip()]

    generator = SyntheticReceiptGenerator()
    print(f"Generating {args.count} synthetic receipts into {out_dir} (noise: {noise_types})...")

    for i in range(1, args.count + 1):
        img, gt = generator.generate(num_items=args.items, noise_types=noise_types)
        img_name = f"receipt_synth_{i:03d}.png"
        gt_name = f"receipt_synth_{i:03d}_truth.json"

        cv2.imwrite(str(out_dir / img_name), img)
        with open(out_dir / gt_name, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

        print(f"  ✓ [{i}/{args.count}] Saved {img_name} ({gt['total_items']} items, {int(gt['total_sum']):,} UZS)")

    print(f"\nDone! Generated {args.count} receipts in {out_dir}")


if __name__ == "__main__":
    main()
