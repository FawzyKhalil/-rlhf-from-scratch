"""Unit tests for data preparation utilities."""

import pytest
from data.prepare_hh_rlhf import split_prompt_response, tokenise


def test_split_prompt_response_basic():
    text = "\n\nHuman: Hello\n\nAssistant: Hi there!"
    prompt, response = split_prompt_response(text)
    assert prompt.endswith("\n\nAssistant:")
    assert response == " Hi there!"


def test_split_prompt_response_multi_turn():
    text = (
        "\n\nHuman: Q1\n\nAssistant: A1"
        "\n\nHuman: Q2\n\nAssistant: A2 final"
    )
    prompt, response = split_prompt_response(text)
    # Should split at the LAST assistant marker
    assert response == " A2 final"
    assert "Q1" in prompt
    assert "Q2" in prompt


def test_split_prompt_response_no_marker():
    text = "some text without any marker"
    prompt, response = split_prompt_response(text)
    assert prompt == text
    assert response == ""


def test_split_returns_full_text_when_concatenated():
    text = "\n\nHuman: test\n\nAssistant: answer"
    prompt, response = split_prompt_response(text)
    assert prompt + response == text


def test_tokenise_truncation():
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    long_prompt = "word " * 300
    response = "response " * 50
    result = tokenise(tokenizer, long_prompt, response, max_length=64)

    assert len(result["input_ids"]) <= 64
    assert len(result["input_ids"]) == len(result["attention_mask"])
    assert all(m == 1 for m in result["attention_mask"])  # no padding from tokenise()


def test_tokenise_short_sequence():
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    result = tokenise(tokenizer, "Hello", " world", max_length=512)
    assert len(result["input_ids"]) < 512
    assert len(result["attention_mask"]) == len(result["input_ids"])
