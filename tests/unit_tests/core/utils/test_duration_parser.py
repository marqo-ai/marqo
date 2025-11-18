"""
Unit tests for duration parser utility.
"""
import unittest
from marqo.core.utils.duration_parser import parse_duration_to_seconds


class TestDurationParser(unittest.TestCase):
    """Test suite for duration string parsing."""

    def test_parse_days_integer(self):
        """Test parsing integer day values."""
        self.assertEqual(parse_duration_to_seconds("1d"), 86400.0)
        self.assertEqual(parse_duration_to_seconds("7d"), 604800.0)
        self.assertEqual(parse_duration_to_seconds("30d"), 2592000.0)

    def test_parse_days_decimal(self):
        """Test parsing decimal day values."""
        self.assertEqual(parse_duration_to_seconds("0.5d"), 43200.0)
        self.assertEqual(parse_duration_to_seconds("1.5d"), 129600.0)
        self.assertEqual(parse_duration_to_seconds("2.25d"), 194400.0)

    def test_parse_hours_integer(self):
        """Test parsing integer hour values."""
        self.assertEqual(parse_duration_to_seconds("1h"), 3600.0)
        self.assertEqual(parse_duration_to_seconds("6h"), 21600.0)
        self.assertEqual(parse_duration_to_seconds("24h"), 86400.0)
        self.assertEqual(parse_duration_to_seconds("168h"), 604800.0)  # 7 days

    def test_parse_hours_decimal(self):
        """Test parsing decimal hour values."""
        self.assertEqual(parse_duration_to_seconds("0.5h"), 1800.0)
        self.assertEqual(parse_duration_to_seconds("1.5h"), 5400.0)
        self.assertEqual(parse_duration_to_seconds("12.5h"), 45000.0)

    def test_parse_zero_duration(self):
        """Test parsing zero duration values."""
        self.assertEqual(parse_duration_to_seconds("0d"), 0.0)
        self.assertEqual(parse_duration_to_seconds("0h"), 0.0)

    def test_parse_large_values(self):
        """Test parsing very large duration values."""
        self.assertEqual(parse_duration_to_seconds("365d"), 31536000.0)  # 1 year
        self.assertEqual(parse_duration_to_seconds("8760h"), 31536000.0)  # 1 year in hours

    def test_invalid_format_missing_unit(self):
        """Test error on duration string missing unit."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("123")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_invalid_format_missing_value(self):
        """Test error on duration string missing numeric value."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("d")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_invalid_unit_minutes(self):
        """Test error on unsupported unit (minutes)."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("30m")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_invalid_unit_seconds(self):
        """Test error on unsupported unit (seconds)."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("60s")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_invalid_unit_milliseconds(self):
        """Test error on unsupported unit (milliseconds)."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("1000ms")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_invalid_unit_uppercase(self):
        """Test error on uppercase units (should be lowercase)."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("1D")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_negative_value(self):
        """Test error on negative duration values."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("-1d")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_non_string_input(self):
        """Test error on non-string input."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds(7)
        self.assertIn("Duration must be a string", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds(7.0)
        self.assertIn("Duration must be a string", str(cm.exception))

    def test_whitespace_not_allowed(self):
        """Test that whitespace in duration string is not allowed."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("1 d")
        self.assertIn("Invalid duration format", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds(" 1d")
        self.assertIn("Invalid duration format", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("1d ")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_empty_string(self):
        """Test error on empty string."""
        with self.assertRaises(ValueError) as cm:
            parse_duration_to_seconds("")
        self.assertIn("Invalid duration format", str(cm.exception))

    def test_equivalence_days_to_hours(self):
        """Test that equivalent durations in different units produce same seconds."""
        # 1 day = 24 hours
        self.assertEqual(
            parse_duration_to_seconds("1d"),
            parse_duration_to_seconds("24h")
        )

        # 7 days = 168 hours
        self.assertEqual(
            parse_duration_to_seconds("7d"),
            parse_duration_to_seconds("168h")
        )

        # 0.5 days = 12 hours
        self.assertEqual(
            parse_duration_to_seconds("0.5d"),
            parse_duration_to_seconds("12h")
        )


if __name__ == '__main__':
    unittest.main()
