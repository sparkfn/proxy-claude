import unittest

try:
    from gateway.proxy_v2.errors import ProxyError
    from gateway.proxy_v2.sse import SSEFrame, SSEParser
except ImportError:
    from proxy_v2.errors import ProxyError
    from proxy_v2.sse import SSEFrame, SSEParser


class ProxyV2SSEParserTests(unittest.TestCase):
    def test_feed_parses_split_sse_frame(self):
        parser = SSEParser()

        frames = parser.feed(b"event: message\n")
        self.assertEqual([], frames)

        frames = parser.feed(b"data: hello\n")
        self.assertEqual([], frames)

        frames = parser.feed(b"data: world\n\n")
        self.assertEqual(
            [SSEFrame(event="message", data="hello\nworld", id=None, retry=None)],
            frames,
        )

    def test_feed_ignores_comments_and_blank_lines(self):
        parser = SSEParser()
        frames = parser.feed(b": keepalive\n\n")
        self.assertEqual([], frames)

    def test_feed_rejects_malformed_field_name(self):
        parser = SSEParser()
        with self.assertRaises(ProxyError):
            parser.feed(b"bad field: value\n\n")

    def test_feed_rejects_unknown_field(self):
        parser = SSEParser()
        with self.assertRaises(ProxyError):
            parser.feed(b"unknown: value\n\n")

    def test_finish_rejects_truncated_event(self):
        parser = SSEParser()
        parser.feed(b"data: partial")
        with self.assertRaises(ProxyError):
            parser.finish()


if __name__ == "__main__":
    unittest.main()
