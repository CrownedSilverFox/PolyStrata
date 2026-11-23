"""Inkling 2 behind the server: its pre-tokenizer, the typed blocks its replies are made of, and a reply that must
not think starting in its text block.

    python -m unittest serve.test_inkling
"""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import (FUNC_START, INK_TEXT_OPEN, ChatTemplate, InklingOutputParser,  # noqa: E402
                            inkling_forced_call)
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

END, STOP = "<|end_message|>", "<|content_model_end_sampling|>"
THINK = "<|message_model|><|content_thinking|>the capital of Italy" + END
TEXT = "<|message_model|><|content_text|>Rome" + END
CALL = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Rome"}}' + END
SCHEMA = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}}}]


def read(reply, step, **kw):
    p = InklingOutputParser(**kw)
    return [e for i in range(0, len(reply), step) for e in p.feed(reply[i:i + step])] + p.finish()


def said(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class PreTokenizer(unittest.TestCase):
    def pieces(self, text):
        return ST.Tokenizer(["a"], [], pre="inkling2")._pieces(text)

    def test_the_pattern_is_gpt_4os(self):
        # digits in threes, a word with the capitals before it, a contraction with its word, ideographs with letters
        self.assertEqual(self.pieces("1234567 dollars"), ["123", "456", "7", " dollars"])
        self.assertEqual(self.pieces("HTTPServer and McDonald's"), ["HTTPServer", " and", " Mc", "Donald's"])
        self.assertEqual(self.pieces("they've  said.\n\n  x"), ["they've", " ", " said", ".\n\n", " ", " x"])
        self.assertEqual(self.pieces("你好ab"), ["你好ab"])
        self.assertEqual(self.pieces("a  \nb"), ["a", "  \n", "b"])

    def test_a_piece_that_is_a_token_is_that_token(self):
        # the model's tokenizer.json sets ignore_merges: "ab" is a token, though no merge makes it.  (In the real
        # vocabulary "kwargs" is such a token: Hugging Face gives [87550], llama.cpp [86226, 70, 82].)
        tok = ST.Tokenizer(["a", "b", "ab"], [], pre="inkling2")
        self.assertTrue(tok.ignore_merges)
        self.assertEqual(tok.encode("ab"), [2])


class Blocks(unittest.TestCase):
    def test_thinking_then_text_however_it_arrives(self):
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = read(THINK + TEXT + STOP, step)
                self.assertEqual((said(evs, "reasoning"), said(evs, "content")), ("the capital of Italy", "Rome"))

    def test_a_call_of_a_declared_tool(self):
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = read(THINK + CALL + STOP, step, tools=SCHEMA)
                calls = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Rome"})])
                self.assertEqual(said(evs, "content"), "")

    def test_a_call_of_a_tool_nobody_declared_is_the_text_it_was_written_as(self):
        other = CALL.replace("get_weather", "rm")
        evs = read(other, 5, tools=SCHEMA)
        self.assertEqual([e.kind for e in evs if e.kind == "tool_call"], [])
        self.assertEqual(said(evs, "content"), other[:-len(END)])

    def test_a_call_that_is_not_json_is_text(self):
        broken = "<|message_model|>get_weather<|content_invoke_tool_json|>{city: Rome" + END
        evs = read(broken, 10_000, tools=SCHEMA)
        self.assertEqual(said(evs, "content"), broken[:-len(END)])

    def test_the_block_end_in_a_string_argument_does_not_end_the_call(self):
        # llama.cpp 29b52d398b reads this as the call; the tags' text can be in a string the model writes as text
        tools = [{"name": "python", "parameters": {"properties": {"code": {"type": "string"}}}}]
        call = '<|message_model|>python<|content_invoke_tool_json|>{"name":"python","args":{"code":"print(\'<|end_message|>\')"}}'
        for step in (1, 4, 10_000):
            with self.subTest(step=step):
                evs = read(call + END + TEXT + STOP, step, tools=tools)
                self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"],
                                 [{"code": "print('<|end_message|>')"}])
                self.assertEqual(said(evs, "content"), "Rome")
        # JSON that never closes: at the end, the block ends at its first tag and is the text it was written as
        broken = '<|message_model|>python<|content_invoke_tool_json|>{"name":"python","args":{"code":"x' + END
        evs = read(broken + TEXT, 3, tools=tools)
        self.assertFalse([e for e in evs if e.kind == "tool_call"])
        self.assertEqual(said(evs, "content"), broken[:-len(END)] + "Rome")

    def test_an_escaped_quote_does_not_end_the_string_that_holds_the_tags(self):
        # PR 211: \" inside the string, then both tags' text, then a second call right after the first
        tools = [{"name": "python", "parameters": {"properties": {"code": {"type": "string"}}}}]
        code = 'print("a \\"' + END + '<|message_model|>\\" b\\\\")'
        call = ('<|message_model|>python<|content_invoke_tool_json|>'
                + json.dumps({"name": "python", "args": {"code": code}}, ensure_ascii=False) + END)
        for step in (1, 5, 10_000):
            with self.subTest(step=step):
                evs = read(call + call + TEXT + STOP, step, tools=tools)
                self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"code": code}] * 2)
                self.assertEqual(said(evs, "content"), "Rome")

    def test_a_reply_whose_text_block_the_prompt_opened(self):
        evs = read("Rome" + END + STOP, 3, opened=True)
        self.assertEqual((said(evs, "reasoning"), said(evs, "content")), ("", "Rome"))

    def test_a_block_cut_off_keeps_what_it_had(self):
        evs = read(THINK + "<|message_model|><|content_text|>Ro", 4)
        self.assertEqual((said(evs, "reasoning"), said(evs, "content")), ("the capital of Italy", "Ro"))

    def test_a_forced_call_opens_the_block_up_to_its_arguments(self):
        opening = inkling_forced_call(FUNC_START + "get_weather>")
        self.assertEqual(opening, '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":')
        self.assertIsNone(inkling_forced_call(None))
        evs = read(opening + '{"city":"Rome"}}' + END, 6, tools=SCHEMA)
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])


class InklingTokens(ByteTokenizer):
    SPECIALS = ["<|message_system|>", "<|message_user|>", "<|message_tool|>", "<|message_model|>", "<|content_text|>",
                "<|content_thinking|>", "<|content_invoke_tool_json|>", END, STOP]
    pre = "inkling2"
    special_ids = {"tokenizer.ggml.eos_token_id": 264}


# Inkling 2's template, cut to what the server depends on: its one switch, the effort, in a system message
TEMPLATE = ('{{- "<|message_system|>effort=" ~ reasoning_effort ~ "<|end_message|>" -}}'
            '{%- for m in messages -%}{{- "<|message_user|>" + m.content + "<|end_message|>" -}}{%- endfor -%}')


class ServiceForInkling(unittest.TestCase):
    def setUp(self):
        self.tok = InklingTokens()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(TEMPLATE, encoding="utf-8")
            self.svc = Service(MockEngine(self.tok, "hi"), self.tok, ChatTemplate(path))
        self.chat = [{"role": "user", "content": "capital of Italy?"}]

    def test_its_blocks_are_read_by_its_own_parser(self):
        self.assertEqual(self.svc.call_format, "inkling")
        self.assertIs(self.svc.parser_cls, InklingOutputParser)
        self.assertEqual(self.svc.text_open_ids, [259, 260])    # <|message_model|><|content_text|>
        self.assertIn(264, self.svc.stop_ids)

    def test_the_effort_the_template_is_given(self):
        for kwargs, effort in (({}, "high"), ({"reasoning_effort": "low"}, "low"), ({"reasoning_effort": 0.5}, "0.5"),
                               ({"reasoning_effort": "ludicrous"}, "high"), ({"enable_thinking": False}, "none")):
            with self.subTest(**kwargs):
                text = self.svc.render_prompt(self.chat, None, kwargs)
                self.assertTrue(text.startswith(f"<|message_system|>effort={effort}<|end_message|>"), text)
                self.assertEqual(text.endswith(INK_TEXT_OPEN), kwargs.get("enable_thinking") is False)

    def test_a_shared_prefix_is_the_first_messages_whatever_the_effort(self):
        # strata_prefix renders the first messages with the effort render_prompt writes: the common start is the
        # whole first message, not the few tokens before the effort line differs
        msgs = [{"role": "user", "content": "DOC " * 40}, {"role": "user", "content": "Which word repeats?"}]
        for kwargs in ({}, {"reasoning_effort": "low"}, {"enable_thinking": False}, {"reasoning_effort": "ludicrous"}):
            with self.subTest(**kwargs):
                req = {"strata_prefix": {"messages": 1}}
                ids, _, _ = self.svc.prepare(msgs, None, dict(kwargs), 16, req=req)
                head = self.tok.decode(ids[:req["strata_prefix"].get("tokens", 0)])
                self.assertTrue(head.endswith("DOC <|end_message|>"), head[-60:])

    def test_a_reply_goes_on_in_the_text_block_a_prompt_opened(self):
        for kwargs, reply, reasoning in (({}, THINK + TEXT + STOP, "the capital of Italy"),
                                         ({"enable_thinking": False}, "Rome" + END + STOP, "")):
            with self.subTest(**kwargs):
                ids = self.svc.encode_prompt(self.chat, None, kwargs)
                self.svc.engine = MockEngine(self.tok, reply)
                evs = [e for kind, e in self.svc.run(ids, not kwargs, None, 64, {}, threading.Event())
                       if kind == "event"]
                self.assertEqual((said(evs, "reasoning"), said(evs, "content")), (reasoning, "Rome"))

    def test_a_named_tool_choice_is_opened_once_the_thinking_is_closed(self):
        # the model thinks; the server writes the call's opening right after the closed thinking block, in place of
        # the text block the model would open, and the model goes on with the arguments
        tools = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}}}]
        force = self.svc.force_text(FUNC_START + "get_weather>\n")
        ids = self.svc.encode_prompt(self.chat, tools, {})
        self.svc.engine = MockEngine(self.tok, [THINK + "<|message_model|><|content_text|>No tool needed.",
                                                '{"city":"Rome"}}' + END + STOP])
        evs = [e for kind, e in self.svc.run(ids, True, tools, 256, {}, threading.Event(), force) if kind == "event"]
        forced = self.tok.encode(force, parse_special=True)
        self.assertEqual(self.svc.engine.last_prompt[-len(forced):], forced)
        self.assertEqual(self.tok.decode(self.svc.engine.last_prompt[len(ids):]), THINK + force)
        self.assertEqual(said(evs, "reasoning"), "the capital of Italy")
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])
        self.assertEqual(said(evs, "content"), "")

    def test_the_parser_says_where_a_forced_call_can_go(self):
        p = InklingOutputParser(tools=SCHEMA)
        p.feed("<|message_model|><|content_thinking|>x")
        self.assertEqual(p.state, "reasoning")
        p.feed(END)
        self.assertEqual((p.state, p.clean), ("content", True))     # between blocks
        p.feed("<|message_model|><|content_text|>y")
        self.assertEqual(p.state, "text")                            # not inside a text block

    def test_reasoning_tokens_are_the_thinking_block_s(self):
        # PR 212: the text block's tokens are not thinking tokens (usage's reasoning_tokens)
        ids = self.svc.encode_prompt(self.chat, None, {})
        self.svc.engine = MockEngine(self.tok, THINK + TEXT + STOP)
        done = [e for kind, e in self.svc.run(ids, True, None, 64, {}, threading.Event()) if kind == "done"][0]
        self.assertEqual(done["reasoning_tokens"], len(self.tok.encode("the capital of Italy" + END, parse_special=True)))


if __name__ == "__main__":
    unittest.main()
