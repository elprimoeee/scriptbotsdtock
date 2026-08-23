from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.trading212_client import (
    DEMO_BASE_URL,
    Trading212Credentials,
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

    def test_market_orders_are_not_retried(self) -> None:
        client = Trading212PracticeClient(Trading212Credentials("abc", "xyz"))
        with patch.object(client, "_request", return_value={"id": 123}) as request:
            result = client.market_order(ticker="AAPL_US_EQ", quantity=0.1)
        self.assertEqual(result["id"], 123)
        request.assert_called_once_with(
            "POST",
            "/api/v0/equity/orders/market",
            payload={"ticker": "AAPL_US_EQ", "quantity": 0.1, "extendedHours": False},
            retry_gets=0,
        )


if __name__ == "__main__":
    unittest.main()
