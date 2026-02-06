"""Unit tests for search.tokenizer."""

import pytest
from search.tokenizer import tokenize


class TestTokenizeASCII:
    def test_simple_words(self):
        tokens = tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_camel_case_split(self):
        tokens = tokenize("getUserName")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_upper_camel_case(self):
        tokens = tokenize("HTMLParser")
        assert "html" in tokens
        assert "parser" in tokens

    def test_snake_case_split(self):
        tokens = tokenize("get_user_name")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_single_char_ascii_removed(self):
        tokens = tokenize("a b c hello")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "hello" in tokens

    def test_stop_words_removed(self):
        tokens = tokenize("the function is not working")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "not" not in tokens
        assert "function" in tokens
        assert "working" in tokens

    def test_numbers_preserved(self):
        tokens = tokenize("http2 error404")
        assert "http2" in tokens
        assert "error404" in tokens


class TestTokenizeCJK:
    def test_kanji_bigram(self):
        tokens = tokenize("契約期間")
        # Should have unigrams
        assert "契" in tokens
        assert "約" in tokens
        assert "期" in tokens
        assert "間" in tokens
        # Should have bigrams
        assert "契約" in tokens
        assert "約期" in tokens
        assert "期間" in tokens

    def test_hiragana(self):
        tokens = tokenize("こんにちは")
        assert "こん" in tokens
        assert "んに" in tokens

    def test_katakana(self):
        tokens = tokenize("パーサー")
        assert "パー" in tokens
        assert "ーサ" in tokens

    def test_single_cjk_char(self):
        tokens = tokenize("木")
        assert "木" in tokens
        assert len(tokens) == 1  # Just the unigram, no bigram possible


class TestTokenizeMixed:
    def test_code_with_japanese_comment(self):
        text = "def calculate_total():  # 合計を計算する"
        tokens = tokenize(text)
        assert "calculate" in tokens
        assert "total" in tokens
        assert "合計" in tokens
        assert "計算" in tokens

    def test_class_name_and_cjk(self):
        text = "UserManager ユーザー管理"
        tokens = tokenize(text)
        assert "user" in tokens
        assert "manager" in tokens
        assert "ユー" in tokens
        assert "管理" in tokens


class TestTokenizeEdgeCases:
    def test_empty_string(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("   \t\n  ") == []

    def test_only_stop_words(self):
        assert tokenize("the is a") == []

    def test_only_single_chars(self):
        assert tokenize("a b c") == []

    def test_special_characters(self):
        tokens = tokenize("hello@world.com foo-bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "com" in tokens
        assert "foo" in tokens
        assert "bar" in tokens
