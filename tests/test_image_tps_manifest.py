"""Focused checks for the first read-only image/TPS manifest."""

import csv
import tempfile
import unittest
from pathlib import Path

from image_tps_manifest import MANIFEST_COLUMNS, build_manifest, write_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.tps = self.root / "tps"
        self.images.mkdir()
        self.tps.mkdir()
        self.specimens = self.root / "specimens.csv"

    def write_specimens(self, rows):
        with self.specimens.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("specimen_id", "group", "split", "image_file", "tps_file", "scale_value", "scale_unit"),
            )
            writer.writeheader()
            writer.writerows(rows)

    def specimen(self, specimen_id="sp01", image_file="sp01.png", tps_file=""):
        return {
            "specimen_id": specimen_id,
            "group": "eastern",
            "split": "train",
            "image_file": image_file,
            "tps_file": tps_file,
            "scale_value": "",
            "scale_unit": "",
        }

    def test_valid_image_tps_pair_is_read_only(self):
        image = self.images / "sp01.png"
        image.write_bytes(b"example image bytes")
        tps = self.tps / "sp01.tps"
        tps.write_text("LM=3\n1 2\n3 4\n5 6\nIMAGE=sp01.png\nSCALE=2.5\n", encoding="utf-8")
        specimen = self.specimen(tps_file="sp01.tps")
        specimen.update(scale_value="2.5", scale_unit="mm")
        self.write_specimens([specimen])
        before = (image.read_bytes(), tps.read_bytes(), self.specimens.read_bytes())

        rows = build_manifest(self.images, self.specimens, self.tps, expected_landmarks=3)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["check_status"], "ok")
        self.assertEqual(rows[0]["landmark_count"], "3")
        self.assertEqual(rows[0]["scale_value"], "2.5")
        self.assertEqual(rows[0]["scale_unit"], "mm")
        self.assertEqual(rows[0]["image_path"], str(image.resolve()))
        self.assertEqual(rows[0]["tps_path"], str(tps.resolve()))
        self.assertEqual(before, (image.read_bytes(), tps.read_bytes(), self.specimens.read_bytes()))
        self.assertEqual(sorted(self.root.iterdir()), sorted((self.images, self.tps, self.specimens)))

    def test_missing_image_keeps_expected_specimen_row(self):
        self.write_specimens([self.specimen()])

        rows = build_manifest(self.images, self.specimens)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["specimen_id"], "sp01")
        self.assertEqual(rows[0]["check_status"], "error")
        self.assertIn("missing_image", rows[0]["check_details"])

    def test_repeated_specimen_id_flags_every_repeated_row(self):
        (self.images / "sp01.png").write_bytes(b"image one")
        (self.images / "sp01.jpg").write_bytes(b"image two")
        self.write_specimens(
            [self.specimen(image_file="sp01.png"), self.specimen(specimen_id="SP01", image_file="sp01.jpg")]
        )

        rows = build_manifest(self.images, self.specimens)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["check_status"] == "error" for row in rows))
        self.assertTrue(all("duplicate_specimen_id" in row["check_details"] for row in rows))

    def test_wrong_number_of_landmarks_is_reported(self):
        (self.images / "sp01.png").write_bytes(b"image")
        (self.tps / "sp01.tps").write_text("LM=3\n1 2\n3 4\nIMAGE=sp01.png\n", encoding="utf-8")
        self.write_specimens([self.specimen(tps_file="sp01.tps")])

        rows = build_manifest(self.images, self.specimens, self.tps, expected_landmarks=3)

        self.assertEqual(rows[0]["landmark_count"], "2")
        self.assertEqual(rows[0]["check_status"], "error")
        self.assertIn("tps_landmark_count_mismatch", rows[0]["check_details"])
        self.assertIn("unexpected_landmark_count", rows[0]["check_details"])

    def test_written_manifest_has_agreed_columns_and_refuses_overwrite(self):
        (self.images / "sp01.png").write_bytes(b"image")
        self.write_specimens([self.specimen()])
        output = self.root / "manifest.csv"
        rows = build_manifest(self.images, self.specimens)

        write_manifest(rows, output)
        with output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames), MANIFEST_COLUMNS)
            self.assertEqual(next(reader)["check_status"], "ok")
        with self.assertRaises(FileExistsError):
            write_manifest(rows, output)


if __name__ == "__main__":
    unittest.main()
