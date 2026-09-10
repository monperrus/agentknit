"""Every HTTP completion must carry X-Title: agentknit.

The SuperleanAI gateway classifies requests by client using X-Title /
User-Agent; without this header agentknit traffic lands in "unknown".
"""

from __future__ import annotations

import unittest
from unittest import mock

from agentknit.openai_compat import OpenAI


class TestXTitleHeader(unittest.TestCase):
    def test_build_url_and_headers_sets_x_title(self) -> None:
        client = OpenAI(api_key="sk-test", base_url="http://localhost:8000")
        _url, headers = client.chat.completions._build_url_and_headers()
        self.assertEqual(headers.get("X-Title"), "agentknit")
        self.assertEqual(headers.get("Authorization"), "Bearer sk-test")

    def test_extra_headers_can_override_x_title(self) -> None:
        client = OpenAI(api_key="sk-test", base_url="http://localhost:8000",
                        extra_headers={"X-Title": "custom-label"})
        _url, headers = client.chat.completions._build_url_and_headers()
        self.assertEqual(headers.get("X-Title"), "custom-label")


if __name__ == "__main__":
    unittest.main()
