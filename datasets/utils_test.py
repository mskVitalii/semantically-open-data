import unittest
from utils import sanitize_filename


class TestSanitizeFilename(unittest.TestCase):
    def test_remove_invalid_chars(self):
        """Test removal of invalid characters"""
        # All invalid characters
        filename = 'test<>:"/\\|?* file.txt'
        expected = "test__________file.txt"
        self.assertEqual(sanitize_filename(filename), expected)

        # Individual invalid characters
        self.assertEqual(sanitize_filename("test<file.txt"), "test_file.txt")
        self.assertEqual(sanitize_filename("test>file.txt"), "test_file.txt")
        self.assertEqual(sanitize_filename("test:file.txt"), "test_file.txt")
        self.assertEqual(sanitize_filename('test"file.txt'), "test_file.txt")
        self.assertEqual(sanitize_filename("test/file.txt"), "test_file.txt")
        self.assertEqual(sanitize_filename("test\\file.txt"), "test_file.txt")
        self.assertEqual(sanitize_filename("test|file.txt"), "test_file.txt")
        self.assertEqual(sanitize_filename("test?file.txt"), "test_file.txt")
        self.assertEqual(sanitize_filename("test*file.txt"), "test_file.txt")
        self.assertEqual(sanitize_filename("test file.txt"), "test_file.txt")

    def test_valid_filename(self):
        """Test with valid filename"""
        filename = "valid_filename.txt"
        self.assertEqual(sanitize_filename(filename), filename)

        filename = "file-name_123.pdf"
        self.assertEqual(sanitize_filename(filename), filename)

    def test_length_limit(self):
        """Test length limitation"""
        # Filename with exactly 200 characters
        filename = "a" * 200
        self.assertEqual(sanitize_filename(filename), filename)
        self.assertEqual(len(sanitize_filename(filename)), 200)

        # Filename longer than 200 characters
        filename = "a" * 250
        result = sanitize_filename(filename)
        self.assertEqual(result, "a" * 200)
        self.assertEqual(len(result), 200)

        # Filename with invalid characters and exceeding length
        filename = "test<>:" + "a" * 250
        result = sanitize_filename(filename)
        expected = "test___" + "a" * 193  # 7 characters replaced + 193 'a' = 200
        self.assertEqual(result, expected)
        self.assertEqual(len(result), 200)

    def test_empty_string(self):
        """Test with empty string"""
        self.assertEqual(sanitize_filename(""), "")

    def test_only_invalid_chars(self):
        """Test string containing only invalid characters"""
        filename = '<>:"/\\|?* '
        expected = "__________"
        self.assertEqual(sanitize_filename(filename), expected)

    def test_multiple_consecutive_invalid_chars(self):
        """Test multiple consecutive invalid characters"""
        filename = "test<<<>>>file.txt"
        expected = "test______file.txt"
        self.assertEqual(sanitize_filename(filename), expected)

    def test_unicode_characters(self):
        """Test with Unicode characters"""
        filename = "file_with_unicode_文件.txt"
        self.assertEqual(sanitize_filename(filename), filename)

        filename = "file:with*invalid?chars.txt"
        expected = "file_with_invalid_chars.txt"
        self.assertEqual(sanitize_filename(filename), expected)

    def test_special_filenames(self):
        """Test special filenames"""
        # Unix hidden files
        self.assertEqual(sanitize_filename(".hidden"), ".hidden")
        self.assertEqual(sanitize_filename(".hidden:file"), ".hidden_file")

        # Files without extension
        self.assertEqual(sanitize_filename("README"), "README")
        self.assertEqual(sanitize_filename("README*"), "README_")


if __name__ == "__main__":
    unittest.main()
