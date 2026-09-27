from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from models.trading212_client import (
    DEMO_BASE_URL,
    Trading212Credentials,
    Trading212Error,
    Trading212PracticeClient,
)


class Trading212ClientTests(unittest.TestCase):
    def test_numeric_leading_env_names_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("212_API_KEY_ID=abc\n212_API_KEY_SECRET=xyz\n", encoding="utf-8")
            credentials = Trading212Credentials.from_env_file(path)
        self.assertEqual(credentials.api_key, "abc")
        self.assertEqual(credentials.api_secret, "xyz")

    def test_non_demo_host_is_rejected(self) -> None:
        credentials = Trading212Credentials("abc", "xyz")
        with self.assertRaisesRegex(ValueError, "locked"):
            Trading212PracticeClient(credentials, base_url="https://live.trading212.com")

    @patch("models.trading212_client.urlopen")
    def test_market_order_posts_once_to_demo(self, urlopen: MagicMock) -> None:
        response = MagicMock()
        response.read.return_value = b'{"id": 42, "status": "NEW"}'
        response.__enter__.return_value = response
        urlopen.return_value = response

        client = Trading212PracticeClient(Trading212Credentials("abc", "xyz"))
        order = client.market_order(ticker="AAPL_US_EQ", quantity=0.01)

        self.assertEqual(order["id"], 42)
        self.assertEqual(urlopen.call_count, 1)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, f"{DEMO_BASE_URL}/api/v0/equity/orders/market")
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {"ticker": "AAPL_US_EQ", "quantity": 0.01, "extendedHours": False},
        )

    @patch("models.trading212_client.urlopen", side_effect=URLError("timeout"))
    def test_market_order_timeout_is_not_retried(self, urlopen: MagicMock) -> None:
        client = Trading212PracticeClient(Trading212Credentials("abc", "xyz"))
        with self.assertRaisesRegex(Trading212Error, "outcome may be unknown"):
            client.market_order(ticker="AAPL_US_EQ", quantity=0.01)
        self.assertEqual(urlopen.call_count, 1)

if __name__ == "__main__":
    unittest.main()
