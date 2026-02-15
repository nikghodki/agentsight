"""Tests for payload hygiene (redaction, truncation, filtering)."""

import pytest

from agent_observability.redaction import PayloadPolicy, sanitize_attributes


class TestSanitizeAttributes:
    def test_passthrough_safe_values(self):
        policy = PayloadPolicy()
        attrs = {"count": 42, "name": "hello", "active": True, "rate": 0.95}
        result = sanitize_attributes(attrs, policy)
        assert result == attrs

    def test_redact_by_key(self):
        policy = PayloadPolicy()
        attrs = {"username": "alice", "password": "s3cret", "api_key": "abc123"}
        result = sanitize_attributes(attrs, policy)
        assert result["username"] == "alice"
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"

    def test_redact_by_pattern_in_value(self):
        policy = PayloadPolicy()
        attrs = {"config": "password=hunter2 host=localhost"}
        result = sanitize_attributes(attrs, policy)
        assert "hunter2" not in result["config"]
        assert "[REDACTED]" in result["config"]

    def test_truncate_long_strings(self):
        policy = PayloadPolicy(max_str_len=50)
        attrs = {"data": "x" * 200}
        result = sanitize_attributes(attrs, policy)
        assert len(result["data"]) < 200
        assert "truncated" in result["data"]

    def test_max_attr_count(self):
        policy = PayloadPolicy(max_attr_count=3)
        attrs = {f"key_{i}": i for i in range(10)}
        result = sanitize_attributes(attrs, policy)
        assert len(result) == 3

    def test_drop_keys(self):
        policy = PayloadPolicy(drop_keys={"internal_debug"})
        attrs = {"name": "test", "internal_debug": "verbose data"}
        result = sanitize_attributes(attrs, policy)
        assert "internal_debug" not in result
        assert result["name"] == "test"

    def test_allow_keys_mode(self):
        policy = PayloadPolicy(allow_keys={"name", "count"})
        attrs = {"name": "test", "count": 5, "extra": "should be dropped"}
        result = sanitize_attributes(attrs, policy)
        assert "name" in result
        assert "count" in result
        assert "extra" not in result

    def test_nested_dict_sanitization(self):
        policy = PayloadPolicy()
        attrs = {"config": {"password": "secret", "host": "localhost"}}
        result = sanitize_attributes(attrs, policy)
        assert result["config"]["password"] == "[REDACTED]"
        assert result["config"]["host"] == "localhost"

    def test_list_sanitization(self):
        policy = PayloadPolicy()
        attrs = {"items": ["safe", "api_key=abc123"]}
        result = sanitize_attributes(attrs, policy)
        assert result["items"][0] == "safe"
        assert "[REDACTED]" in result["items"][1]

    def test_does_not_mutate_input(self):
        policy = PayloadPolicy()
        attrs = {"password": "secret", "name": "test"}
        original_password = attrs["password"]
        sanitize_attributes(attrs, policy)
        assert attrs["password"] == original_password

    def test_ssn_redaction(self):
        policy = PayloadPolicy()
        attrs = {"note": "SSN is 123-45-6789 for the record"}
        result = sanitize_attributes(attrs, policy)
        assert "123-45-6789" not in result["note"]

    def test_credit_card_redaction(self):
        policy = PayloadPolicy()
        attrs = {"payment": "Card 4111-1111-1111-1111 on file"}
        result = sanitize_attributes(attrs, policy)
        assert "4111-1111-1111-1111" not in result["payment"]

    def test_bearer_token_redaction(self):
        policy = PayloadPolicy()
        attrs = {"header": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"}
        result = sanitize_attributes(attrs, policy)
        assert "eyJhbGci" not in result["header"]
