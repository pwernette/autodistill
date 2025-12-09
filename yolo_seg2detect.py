import os
from pathlib import Path
from typing import List

class YOLOSegToDetConverter:
    """
    Converts a YOLO segmentation dataset (with polygon coordinates)
    into a YOLO detection dataset (bounding boxes only).
    """

    def __init__(self, input_root: str, output_root: str, verbose: bool = True):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.verbose = verbose

        if not self.input_root.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)

    def convert(self):
        label_files = list(self.input_root.rglob("*.txt"))
        if self.verbose:
            print(f"Found {len(label_files)} label files to process.")

        for label_path in label_files:
            rel_path = label_path.relative_to(self.input_root)
            output_label_path = self.output_root / rel_path
            output_label_path.parent.mkdir(parents=True, exist_ok=True)

            det_lines = self._convert_label_file(label_path)
            if det_lines:
                output_label_path.write_text("\n".join(det_lines) + "\n")
            elif self.verbose:
                print(f"⚠️ Skipped empty or invalid file: {label_path}")

        if self.verbose:
            print(f"✅ Conversion complete! Detection labels saved to: {self.output_root}")

    def _convert_label_file(self, label_path: Path) -> List[str]:
        """Read a YOLO segmentation label file and extract detection lines."""
        det_lines = []

        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < 5:
                    if self.verbose:
                        print(f"  ⚠️ Skipping malformed line {line_num} in {label_path}")
                    continue

                # Keep only class_id + bbox (first 5 entries)
                cls, x, y, w, h = parts[:5]
                try:
                    # Validate numeric values are between 0 and 1
                    vals = [float(v) for v in (x, y, w, h)]
                    if not all(0 <= v <= 1 for v in vals):
                        raise ValueError
                    det_lines.append(" ".join([cls, x, y, w, h]))
                except ValueError:
                    if self.verbose:
                        print(f"  ⚠️ Invalid bbox values at line {line_num} in {label_path}")
                    continue

        return det_lines


if __name__ == "__main__":
    # Example usage:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert YOLO segmentation dataset to YOLO detection format."
    )
    parser.add_argument("-in","-input",
                        dest="input",
                        default='/mnt/d/sfm_greeen_2025harvest/20250926_oxley/photos_geotagged_3/Training_Data/GrapeMapper_segment_0.5_0.5_0.01_0.3_JPG',
                        # required=True, 
                        help="Path to segmentation labels root.")
    parser.add_argument("-output", '-out',
                        default='/mnt/d/sfm_greeen_2025harvest/20250926_oxley/photos_geotagged_3/Training_Data/GrapeMapper_detect_0.5_0.5_0.01_0.3_JPG',
                        # required=True, 
                        help="Path to save detection labels.")
    parser.add_argument("-quiet", 
                        action="store_true", 
                        help="Suppress verbose output.")
    args = parser.parse_args()

    converter = YOLOSegToDetConverter(
        input_root=args.input,
        output_root=args.output,
        verbose=not args.quiet
    )
    converter.convert()
