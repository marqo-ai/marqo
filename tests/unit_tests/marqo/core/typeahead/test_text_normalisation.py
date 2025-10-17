import unittest
import unicodedata
from hypothesis import given, strategies as st

from marqo.core.typeahead.text_normalization import normalize_text, generate_prefixes


class TestNormalizeText(unittest.TestCase):

    # ---------- Basic behavior ----------
    def test_empty_string_returns_empty(self):
        self.assertEqual(normalize_text(""), "")

    def test_none_returns_empty(self):
        # Accepts None defensively (your function does a truthiness check)
        self.assertEqual(normalize_text(None), "")  # type: ignore[arg-type]

    def test_basic_lowercasing(self):
        self.assertEqual(normalize_text("ABC xyz"), "abc xyz")

    def test_punctuation_and_whitespace_preserved(self):
        self.assertEqual(normalize_text("  Résumé!  "), "  resume!  ")

    # ---------- Accents & compatibility ----------
    def test_accent_removal_and_lowercase(self):
        cases = [
            ("résumé", "resume"),
            ("Café", "cafe"),
            ("mañana", "manana"),
            ("Māori", "maori"),
            ("e\u0301lite", "elite"),   # decomposed "é"
            ("İstanbul", "istanbul"),   # I + combining dot -> remove dot -> i
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), expected)

    def test_ligatures_expand_under_nfkd(self):
        # NFKD expands common Latin ligatures
        s = "ﬀ ﬁ ﬂ ﬃ ﬄ"  # ff, fi, fl, ffi, ffl
        self.assertEqual(normalize_text(s), "ff fi fl ffi ffl")

    def test_superscripts_compatibility_decompose(self):
        # NFKD converts many superscripts to base digits
        self.assertEqual(normalize_text("x² + y³ = z⁴"), "x2 + y3 = z4")

    # ---------- Non-Latin scripts ----------
    def test_greek_tonos_removed(self):
        # Αθήνα -> αθηνα
        self.assertEqual(normalize_text("Αθήνα"), "αθηνα")

    def test_cyrillic_breve_removed(self):
        # Й -> И + combining breve -> remove -> и
        self.assertEqual(normalize_text("Й"), "и")

    # ---------- Things that should NOT change ----------
    def test_emoji_are_preserved(self):
        self.assertEqual(normalize_text("Shoes 👟👟"), "shoes 👟👟")

    def test_german_sharp_s_not_mapped_to_ss(self):
        # Using lower(), not casefold(), so ß remains ß
        self.assertEqual(normalize_text("Straße"), "straße")

    # ---------- Idempotency & invariants ----------
    def test_idempotent(self):
        samples = [
            "",
            "Résumé!",
            "CAFÉ du Cycliste",
            "H&M 2-Pack Socks",
            "İstanbul — 2025",
            "Αθήνα & Αθήνα",
            "x² + y³",
        ]
        for s in samples:
            with self.subTest(s=s):
                once = normalize_text(s)
                twice = normalize_text(once)
                self.assertEqual(once, twice)

    @given(st.text())
    def test_no_combining_marks_remain(self, random_text):
        out = normalize_text(random_text)
        self.assertTrue(all(unicodedata.combining(ch) == 0 for ch in out))

    @given(st.text())
    def test_output_is_lowercase(self, random_text):
        out = normalize_text(random_text)
        self.assertEqual(out, out.lower())

    @given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
    def test_ascii_input_behaves_like_lower(self, random_ascii):
        # For printable ASCII, normalization ~= lowercasing
        self.assertEqual(normalize_text(random_ascii), random_ascii.lower())


class TestGeneratePrefixes(unittest.TestCase):
    def test_generate_prefixes_for_single_term_query(self):
        query = 'hello'
        prefixes = generate_prefixes(query)
        self.assertListEqual(['h', 'he', 'hel', 'hell', 'hello'], prefixes)

    def test_generate_prefixes_for_multi_term_query(self):
        query = 'hello world'
        prefixes = generate_prefixes(query)
        self.assertListEqual(['h', 'he', 'hel', 'hell', 'hello', 'w', 'wo', 'wor', 'worl', 'world'], prefixes)

    def test_generate_prefixes_for_multi_term_query_with_multiple_spaces(self):
        query = '   hello    world    '
        prefixes = generate_prefixes(query)
        self.assertListEqual(['h', 'he', 'hel', 'hell', 'hello', 'w', 'wo', 'wor', 'worl', 'world'], prefixes)


if __name__ == "__main__":
    unittest.main()
