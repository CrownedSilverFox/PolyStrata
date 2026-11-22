# Inkling 2 Mini

Back to [README.md](../../README.md)

**Inkling 2 Mini** by Thinking Machines: a file of 9 to 18 GB. It is experimental here. It has run on Linux with an
RTX 5090 and on Windows with an 8 GB laptop card, text only, one request at a time. Its 24 layers of weights run as 48: the twelve in the middle run three
times for every token. Forty of the 48 look back 1,024 tokens only, so a long conversation takes little VRAM:
32,768 bytes a token.

## How fast is it?

Measured on an RTX 5090 (32 GB) with a Xeon Platinum 8470Q (25 vCPU) and 90 GB of RAM, on Linux. The card holds
every expert of all three sizes. Each run writes 256 tokens, the most likely one each time, at a 32K context, and each
number is the range of three runs.

| Size | Writes after a 19-token prompt | Writes after a 10,081-token prompt | Reads that prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 111-112 tokens/s | 109 tokens/s | 3,290-3,300 tokens/s |
| **UD-Q3_K_XL** | 114 tokens/s | 110-112 tokens/s | 3,360-3,450 tokens/s |

llama.cpp on the same card, run in turns with the above on the same prompts (commit 29b52d3 with the fix of its
pull request 32704, `llama-server -ngl 99`):

| Size | Writes after a 19-token prompt | Writes after a 10,081-token prompt | Reads that prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 82-83 tokens/s | 76-77 tokens/s | 1,540-1,550 tokens/s |
| **UD-Q3_K_XL** | 84-86 tokens/s | 77-78 tokens/s | 1,510-1,530 tokens/s |

Both runs print the three likeliest tokens of every step, which the comparison below reads; that costs a little.
Without it, UD-Q4_K_XL writes 127-129 tokens/s after the short prompt and 124 after the long one, and reads the long
one at 3,280-3,320 tokens/s (six runs each); the README's benchmark measures 128.

The two start with the same tokens. For UD-Q4_K_XL the first 13 after the short prompt and the first 14 after the
long one are the same, and they part at a step whose two leading candidates are nearly as likely (0.35 against 0.34,
and 0.028 against 0.028). For UD-Q3_K_XL the first 13 and the first 40 are the same.

On a laptop with an RTX 4070 Laptop (8 GB), a Core i7-14700HX and 32 GB of RAM, on Windows, UD-Q3_K_XL has 1,075 of
its 3,072 experts on the card and the rest are computed on the CPU. Three runs of the README's benchmark:

| Size | Writes in a short chat | Reads a 30,954-token prompt | Writes after it |
| --- | ---: | ---: | ---: |
| **UD-Q3_K_XL** | 34-37 tokens/s | 415-417 tokens/s | 28-29 tokens/s |

Reading is the slow part on a small card. The model's looped layers make every batch of the prompt upload the
experts that are not on the card again. With `--prompt-batch 2048` the same prompt was read at 539 tokens/s (one
run).

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** card with **8 GB of VRAM or more**. It has run on an RTX 5090 (32 GB) and on an RTX 4070 Laptop (8 GB). About 3.7 GB are taken before any expert at a 32K context. |
| **RAM** | About **21, 17 or 13 GB** beside an 8 GB card for UD-Q4_K_XL, UD-Q3_K_XL and UD-Q2_K_XL, the installer's estimate. A 32 GB card holds any of them whole. |
| **Disk** | 9.1 to 17.6 GB for the model file. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver. |

Not run yet: an AMD card, a Mac, several cards, and UD-Q4_K_XL on the 8 GB card. The model also reads
pictures and has a draft file for faster answers; neither is used here yet. The engine takes one request at a time
for this model, and the installer writes `"parallel": 1`. Its license is Apache 2.0.

## Install

```sh
./setup.sh --family inkling
```

Add `--model UD-Q3_K_XL` or `--model UD-Q2_K_XL` for a smaller size.

## Which model should I pick?

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | 4-bit experts, 8-bit where they write ([Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)) | 17.6 GB | about 21 GB |
| **UD-Q3_K_XL** | 3- and 4-bit experts mixed | 13.6 GB | about 17 GB |
| **UD-Q2_K_XL** | 1- to 2-bit experts: the smallest | 9.1 GB | about 13 GB |

The installer downloads the files under the repository's `Mainline_Support` folder, the layout llama.cpp runs
(`inkling2_loop`). The files at the top of the repository are another layout, which the engine refuses by name.

## How does it work?

A reply is a list of typed blocks: thinking, text, or a tool call with its arguments as JSON. The server turns
them into the same reasoning, content and tool calls as for the other models, so the OpenAI and Anthropic APIs work
as they do elsewhere. A request that asks for no thinking has its prompt end inside the text block, because the
model otherwise opens a thinking block now and then. More in
[BACKENDS.md](../BACKENDS.md#inkling-2-mini-one-card-text-only).
