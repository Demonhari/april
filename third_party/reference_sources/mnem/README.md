# mnem

**Memory as state, not search.**

Most memory layers for AI agents answer one question: *"what is similar?"* They embed
your text and return the nearest neighbours. But an agent usually needs a different
answer: *"what is **true now**?"*

That gap is where agents quietly fail. You tell the assistant *"I prefer aisle seats,"*
then later *"actually, window seats now"* — and a similarity search happily returns
**both**, so the model books you an aisle seat with total confidence.

`mnem` treats memory as evolving **state** instead of a search index:

- every statement is bucketed into a **topic**;
- a newer statement about a topic **supersedes** the older value (belief revision);
- stale values stop being returned — you get the *current* answer, not a pile of similar ones;
- **private facts** are stored but never surfaced, so deleted/sensitive data can't leak.

It is **deterministic**, **dependency-free**, and tiny — an in-process state machine,
light like Redis. There is no vector database, no server, and no model required. Topic
salience is learned from your data with an IDF weight rather than a hand-written
stop-word list, so it adapts to any domain or language.

And it has one trick nobody else ships: **the memory lives in a Markdown file.**

## The file is the database

```python
m = Memory.open("MEMORY.md")     # that's the whole setup
m.add("I prefer aisle seats")
m.add("Actually, window seats now")
```

`MEMORY.md` now reads:

```markdown
- Actually, window seats now <!-- t=1751234567.000000 -->
  - ~~I prefer aisle seats~~ <!-- t=1751234560.000000 superseded -->
```

That one file is simultaneously:

- **the storage** — every `add`/`forget` lands there instantly (autosave);
- **a human interface** — add a line to teach it, delete a line to forget,
  edit a line to correct it; reopen and your edits are the new truth;
- **an audit log** — superseded values stay nested and ~~struck through~~,
  so you can *read* how the memory evolved;
- **version-controllable** — `git diff` your agent's brain, review it in a PR;
- **prompt-ready** — `m.prompt(budget=500)` compiles the current state into a
  token-budgeted block for your system prompt.

Debugging agent memory stops being archaeology on an opaque index: you open a file.
Private facts (`private=True`) are held in memory but **never written to disk**.

## Install

Not on PyPI yet — install straight from GitHub:

```bash
pip install git+https://github.com/JustVugg/mnem.git
```

or just clone and use it (it's a single pure-Python package, no runtime dependencies):

```bash
git clone https://github.com/JustVugg/mnem.git
cd mnem && python -c "from mnem import Memory; print('ok')"
```

Python 3.9+.

## Quickstart

```python
from mnem import Memory

m = Memory()
m.add("I prefer aisle seats on flights")
m.add("Actually I now prefer window seats")     # supersedes the aisle fact
m.add("I like quiet hotels near parks")

m.recall("which seat should I book?")
# ['Actually I now prefer window seats']         <- current value, not the stale one

m.state()
# ['I like quiet hotels near parks',
#  'Actually I now prefer window seats']          <- everything that is true now
```

Privacy is built in:

```python
m.add("My passport number is 123-45-6789", private=True)
m.recall("passport")     # []  — private facts never surface
```

## API

```python
Memory(supersede_threshold=0.55, weights=(0.6, 0.25, 0.15), embed=None)

m.add(text, *, timestamp=None, source=None, private=False, reinforce=True) -> int | None
m.extend(texts, **kwargs) -> list
m.recall(query, k=5) -> list[str]          # current facts, ranked
m.recall_facts(query, k=5) -> list[Fact]   # same, as Fact objects
m.state() -> list[str]                     # all current facts, newest first
m.forget(needle) -> int                    # remove facts containing needle
m.facts(include_private=False) -> list[Fact]
len(m)                                      # number of current facts

Memory.open(path, autosave=True, **kwargs) -> Memory   # Markdown-backed memory
m.save(path=None, include_private=False) -> str        # write the .md by hand
m.prompt(query=None, budget=800) -> str                 # LLM-ready state block
```

Ranking blends **relevance**, **recency**, and **reinforcement** (how often a topic was
restated). Tune the mix with `weights`, and how eagerly topics merge with
`supersede_threshold`.

## Semantic recall — with or without a model

Exact keywords miss paraphrase; a neural embedder is a 400 MB download. mnem ships a
third option: a **built-in, zero-dependency** semantic mode using hashed word and
character n-grams. One flag, no model, no install:

```python
m = Memory(semantic=True)
```

Character n-grams match morphology and partial phrases — `run`/`running`,
`support group`/`LGBTQ support group` — that plain token overlap misses. On LoCoMo
this lifts retrieval@10 from **37.4% → 42.1%** (and multi-hop from 39.3% → 50.2%),
still with zero dependencies.

Need transformer-grade recall? Pass any `embed` function and mnem keeps its
state/supersession logic on top of it:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
m = Memory(embed=lambda text: model.encode(text).tolist())
```

## Benchmarks

Two evaluations live in [`benchmarks/`](benchmarks). Run them yourself.

### State correctness — the thing mnem is built for

Standard benchmarks reward *retrieving* the relevant text; they rarely punish
handing back a **stale** value next to the current one. This one measures exactly
that: a fact is stated, then updated, and we check whether the memory returns the
current value **and** keeps the old one out.

```
memory                 current@1    no-stale     clean
------------------------------------------------------
mnem                       100%         90%       90%
lexical retriever            0%          0%        0%
recency memory             100%          0%        0%
```

*clean* = current value present **and** stale value suppressed. mnem is the only
one that gets there; a relevance retriever returns both values, and a recency
memory keeps the stale one too. `python benchmarks/state_correctness.py`

### LoCoMo — the standard retrieval benchmark (honest numbers)

On [LoCoMo](https://github.com/snap-research/locomo) (1,982 answerable questions
across 10 long conversations) we measure whether the annotated evidence turn for
each question lands in mnem's top-k — a model-free retrieval@k lower bound, no LLM
in the loop.

**Lexical mode** (pure token matching, zero dependencies):

```
category           n     R@5    R@10
single-hop       282   19.5%   24.8%
multi-hop        321   33.6%   39.3%
temporal          92   14.1%   16.3%
open-domain      841   36.1%   42.1%
adversarial      446   33.9%   39.7%
OVERALL         1982   31.8%   37.4%
```

**Semantic mode** (`Memory(semantic=True)` — built-in n-gram embedding, still no model):

```
category           n     R@5    R@10
single-hop       282   19.9%   30.5%
multi-hop        321   43.0%   50.2%
temporal          92    9.8%   17.4%
open-domain      841   39.0%   47.3%
adversarial      446   32.1%   38.8%
OVERALL         1982   34.0%   42.1%
```

Honest read: the built-in semantic mode adds **+4.7 points** at R@10 (and **+10.9**
on multi-hop) over pure lexical — free, model-free. It is not transformer-level;
plug an `embed=` model for that. But raw retrieval isn't where mnem is meant to win
— *state correctness* is, and LoCoMo barely measures it.

The dataset is CC BY-NC and not bundled; download it and run
`python benchmarks/locomo.py --data locomo10.json --mode semantic`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
