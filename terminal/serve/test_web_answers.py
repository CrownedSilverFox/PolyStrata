"""The web page's answers: ↻ writes another answer and keeps the old one as a variant, the arrows go between them (issue #12,
PR #74). PolyStrata's own, lost once in a merge of upstream's page, so these checks read the sources for it. What it does in a
real browser is in tools/webtest/browser_test.mjs ("regenerate"), run here when WEBTEST_MODULES is set.

    WEBTEST_MODULES=<folder> python -m unittest serve.test_web_answers -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).parent / "web"


class TheSourcesKeepThem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (WEB / "app.js").read_text(encoding="utf-8")

    def test_the_answer_controls_are_there(self):
        for hook in ("data-msg-regen", "data-msg-prev", "data-msg-next", "data-swipe-count"):
            with self.subTest(hook=hook):
                self.assertIn(hook, self.app)
        self.assertIn("generateReply({variant: m});", self.app)

    def test_an_earlier_answer_is_asked_for_without_what_comes_after_it(self):
        self.assertIn("if (busy && busy.msg === m) break;", self.app)


@unittest.skipUnless(os.environ.get("WEBTEST_MODULES") and shutil.which("node"),
                     "set WEBTEST_MODULES (a folder with node_modules/puppeteer-core) to run the browser tests")
class InABrowser(unittest.TestCase):
    def test_answers_in_chrome(self):
        for name in ("regenerate",):
            with self.subTest(name=name):
                r = subprocess.run([shutil.which("node"), str(ROOT / "tools" / "webtest" / "browser_test.mjs"), name],
                                   capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
