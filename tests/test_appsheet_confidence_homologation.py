"""Pruebas de homologación de métricas entre Streamlit y AppSheet."""
import unittest
from appsheet_integration import INVENTORY_HEADERS, _inventory_record_values

class AppSheetConfidenceHomologationTests(unittest.TestCase):
    def test_inventory_exposes_both_confidence_metrics(self):
        self.assertIn("ConfianzaImagen", INVENTORY_HEADERS)
        self.assertIn("ConfianzaSurco", INVENTORY_HEADERS)
        self.assertIn("ConfianzaSlots", INVENTORY_HEADERS)

    def test_image_confidence_uses_streamlit_inventory_average(self):
        row = {
            "id": 1, "slot_count": 10, "vacant_indices": [2, 5],
            "points_norm": [[100, 120], [900, 880]],
            "confidence": 0.80, "slot_confidence": 0.65,
            "row_spacing_px": 18.0, "slot_pitch_px": 9.0,
        }
        record = _inventory_record_values(
            request_id="REQ-TEST", historial_id="HIST-TEST",
            parcela_id="PAR-TEST", parcela_name="Parcela de prueba",
            photo_index=1, source_path="Fotos/test.jpg",
            source_name="test.jpg", inventory_path="Inventarios/test.png",
            row=row, idioma="ES", image_confidence=0.72,
        )
        self.assertEqual(record["ConfianzaSurco"], 80.0)
        self.assertEqual(record["ConfianzaSlots"], 65.0)
        self.assertEqual(record["ConfianzaImagen"], 72.0)
        self.assertEqual(record["Confianza"], 65.0)

    def test_legacy_call_without_image_confidence_remains_compatible(self):
        row = {
            "id": 1, "slot_count": 4, "vacant_indices": [],
            "points_norm": [[0, 0], [1000, 1000]],
            "confidence": 0.9, "slot_confidence": 0.6,
        }
        record = _inventory_record_values(
            request_id="REQ-LEGACY", historial_id="HIST-LEGACY",
            parcela_id="PAR-TEST", parcela_name="Parcela de prueba",
            photo_index=1, source_path="Fotos/test.jpg",
            source_name="test.jpg", inventory_path="Inventarios/test.png",
            row=row, idioma="ES",
        )
        self.assertEqual(record["ConfianzaImagen"], 60.0)

if __name__ == "__main__":
    unittest.main()
