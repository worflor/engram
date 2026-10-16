# The Engram Harness

An `.engram` file is a mind without a body. The harness IS the body.

## The Four Nouns

Everything in the harness is one of four things:

**Brain** — the `.engram` file loaded into memory. Wells, dream buffer, measure, absorb. The geometry. Earned, not given. Implemented in Rust, called via JNI. The brain does NOT declare capabilities — it is the mind, not the body.

**Mouth** — a LOCAL model (ollama). Receives well profile + persona + available tools. Speaks FOR the brain. The brain directs, the mouth articulates. Must be local. No API calls. No cloud. No logs. The mouth sees everything: conversation, wells, personality, tool results. That stays on your machine.

**Hands** — tools. Execute actions in the world. Results feed back into the brain automatically. Tools are data (JSON definitions), not code. Three executor types cover everything. Hands belong to the HARNESS, not the brain. The same brain on a phone has different hands than on a workstation.

**Judges** — other LLMs (API ok) with personality cards as rubrics. Score submitted WORK only. Never see the raw conversation. Never see the well profile. Never see the personality. They see an essay, a code file, a research summary. The diversity of judges IS the value.

```
┌──────────┐
│  Brain   │  .engram file. wells, dream, geometry.
│          │  knows things. can't speak, can't act.
└────┬─────┘
     │
┌────▼─────┐
│  Mouth   │  LOCAL model only. receives well profile + persona.
│          │  speaks FOR the brain. the brain directs.
└────┬─────┘
     │
┌────▼─────┐
│  Hands   │  tools. execute actions. results feed back into brain.
│          │  adding a tool = adding a JSON file.
└────┬─────┘
     │
┌────▼─────┐
│  Judges  │  API models with rubric cards. score submitted work.
│          │  feedback absorbs into wells. she improves.
└──────────┘
```

## The Privacy Boundary

This is architectural, not a policy checkbox.

```
LOCAL (on your machine, always)        REMOTE (opt-in, work only)
─────────────────────────────────      ───────────────────────────
brain (.engram file)                   judges (see submitted work)
mouth (ollama, local inference)        web_search (tool, if enabled)
conversation history
well profiles
personality cards
tool results

the mouth sees EVERYTHING.             judges see ARTIFACTS.
it must stay local.                    an essay. a code file.
no API. no cloud. no logs.             never the conversation.
no terms of service.                   never the wells.
```

## The Protocol

Every interaction is a JSON message in, JSON message out. Same protocol whether you're in a terminal, a React app, or piping from a script. Transport is separate.

```
┌─────────────────────────────────────────────────────────┐
│  chat     { text: "..." }  → { text, drift, well }     │
│  measure  { text: "..." }  → { drift, texture, wells } │
│  absorb   { text, source } → { well_name }             │
│  status   { }              → { wells, dream, stats }   │
│  diff     { a, b }         → { changes }               │
│  train    { task }         → { scores, feedback }       │
└─────────────────────────────────────────────────────────┘
```

### Transports

Three thin adapters over the same protocol:

```
┌───────────┐  ┌───────────┐  ┌───────────┐
│   STDIO   │  │   HTTP    │  │   REPL    │
│           │  │           │  │           │
│ JSON over │  │ REST +    │  │ human-    │
│ stdin/out │  │ WebSocket │  │ friendly  │
│ pipe-     │  │ any web   │  │ terminal  │
│ friendly  │  │ frontend  │  │ chat UI   │
│ LSP-style │  │ mobile    │  │           │
└───────────┘  └───────────┘  └───────────┘
```

### CLI

```
engram chat alexandria.engram              # interactive REPL
engram serve alexandria.engram             # HTTP + WebSocket server
engram measure alexandria.engram < in.txt  # pipe-friendly one-shot
engram train alexandria.engram tasks/      # run task suite
engram diff before.engram after.engram     # what did she learn
engram status alexandria.engram            # well summary
```

### Frontend Examples

Any frontend talks the same protocol:

```js
// TypeScript / React
const ws = new WebSocket("ws://localhost:8401");
ws.send(JSON.stringify({ method: "chat", text: "explain tunneling" }));
ws.onmessage = (e) => {
  const { text, drift, well } = JSON.parse(e.data);
  // drift decreasing across turns = she's converging
};
```

```bash
# pipe from shell
echo '{"method":"measure","text":"..."}' | engram pipe brain.engram
```

```java
// direct library call from Java/Kotlin
var harness = Harness.load("alexandria.engram");
var reply = harness.chat("explain tunneling");
// reply.text, reply.drift, reply.nearestWell
```

## Tools Are Data, Not Code

A tool = name + schema + executor type. Adding a tool = adding a JSON file. No new Java class unless you need a new executor type (almost never).

```
tools/
├── write_file.json
├── read_file.json
├── run_code.json
├── web_search.json
└── bash.json
```

Example tool definition:

```json
{
  "name": "write_file",
  "description": "Write content to a file in the workspace",
  "parameters": {
    "path": { "type": "string", "required": true },
    "content": { "type": "string", "required": true }
  },
  "executor": "file_write",
  "sandbox": true
}
```

```json
{
  "name": "run_code",
  "description": "Execute a command in a sandboxed environment",
  "parameters": {
    "command": { "type": "string", "required": true },
    "timeout": { "type": "integer", "default": 30 }
  },
  "executor": "process",
  "sandbox": true,
  "network": false
}
```

```json
{
  "name": "web_search",
  "description": "Search the web and return results as text",
  "parameters": {
    "query": { "type": "string", "required": true }
  },
  "executor": "http",
  "endpoint": "${SEARCH_ENDPOINT}",
  "sandbox": false
}
```

### Three Executor Types

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   process    │  │    file      │  │    http      │
│              │  │              │  │              │
│ run_code     │  │ read_file    │  │ web_search   │
│ bash         │  │ write_file   │  │ any REST API │
│              │  │              │  │              │
│ Process-     │  │ java.nio     │  │ Http-        │
│ Builder      │  │ (pure Java)  │  │ Client       │
│ sandboxed    │  │ sandboxed    │  │              │
│ temp dir     │  │ to workspace │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

Want a new tool? Write a JSON file. Need a new executor type? THAT'S when you write Java. Almost never.

### The Sandbox

```
each session gets:
  ${tmpdir}/engram/${session-id}/workspace/

process executor:
  - cwd = workspace
  - no network (for run_code)
  - timeout enforced via Process.waitFor(timeout, SECONDS)
  - stdout + stderr captured, returned as result

file executor:
  - all paths resolved relative to workspace
  - Path.normalize() to prevent escape (../../etc/passwd)
  - symlinks not followed outside workspace

the sandbox is invisible. tools just work inside it.
if she writes rm -rf /, it hits the wall.
```

## The Learning Loop

This is automatic. You don't configure it. It just happens.

```
        user says something
              │
              ▼
     ┌────────────────┐
     │ brain.measure() │  where are we in well space?
     │ drift = 34%     │  what do we know? what's new?
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐   well profile + drift + persona
     │ mouth.speak()   │   model responds FROM a location
     └───────┬────────┘   in knowledge space, not the void
             │
      ┌──────┴──────┐
      │             │
 uses a tool    just speaks
      │             │
      ▼             │
┌───────────┐       │
│ hand.do() │       │
│ → result  │       │
└─────┬─────┘       │
      │             │
      ▼             ▼
┌───────────────────────────┐
│     brain.absorb()         │  everything said and found
│                            │  becomes part of the geometry.
│     wells shift.           │
│     drift decreases.       │  next turn, the well profile
│     she knows more.        │  is different. the model sees
│                            │  different ground. responds
│     this is learning.      │  from a new location.
│     no gradient descent.   │
│     no reward model.       │  the conversation has momentum
│     just geometry.         │  because the wells have gravity.
└───────────────────────────┘
```

### What Drift Tells the Mouth

The well profile is the brain's way of giving the model direction:

```
high drift (34%+)     → unfamiliar territory
                        model should be exploratory
                        reach for web_search if available
                        ask clarifying questions

medium drift (10-30%) → partially known
                        model can go deeper
                        connect to nearby wells

low drift (<10%)      → familiar ground
                        model should be precise
                        answer from understanding
                        push past the obvious

drift DECREASING      → converging on understanding
across turns            the conversation has momentum

drift FLAT            → stuck or generic
                        change approach
```

No RLHF needed. Drift decrease IS the reward signal. Curiosity drives tool use. Tool results drive learning. Learning reduces curiosity. Same loop.

## The Judge Committee

Judges are independent LLMs that score submitted WORK (not conversation).

```
alexandria submits work (essay, code, research)
  │
  ├──► judge 1 (opus_strict.json)    ──► score + explain
  ├──► judge 2 (gemini_creative.json) ──► score + explain
  ├──► judge 3 (gpt_technical.json)  ──► score + explain
  │    (parallel, virtual threads, independent)
  │
  │    each judge optionally does certainty rounds:
  │    round 1: read work, score, explain
  │    round 2: re-read + own round 1, "are you sure?"
  │    a judge that changes score between rounds = less certain
  │
  ├──► composing model (cheap, e.g. haiku)
  │    reformats all reviews into normalized JSON
  │    no editorializing. every critique preserved.
  │
  └──► brain.absorb(structured_feedback)
       wells shift based on feedback content
       she tries again with lower drift
```

### Judge Cards

```
judges/
├── strict.json       "demand excellence, zero tolerance for hand-waving"
├── structure.json    "reward coherent structure and logical flow"
├── creative.json     "look for novel connections and creative leaps"
├── technical.json    "verify technical accuracy, catch errors"
└── clarity.json      "can a beginner understand this? simplify."
```

Hot-swappable. Each judge = API endpoint + personality card (JSON). Add new models, remove old ones, no code change.

## Task System

Three tiers. Assignments train. Projects stretch. Checkpoints measure.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Assignments   │  │  Projects    │  │ Checkpoints   │
│ (daily reps)  │  │ (multi-step) │  │ (standardized)│
│               │  │              │  │               │
│ prompt →      │  │ plan →       │  │ same test     │
│ submit →      │  │ research →   │  │ same rubric   │
│ judge →       │  │ draft →      │  │ day 1,7,30    │
│ absorb →      │  │ revise →     │  │ score delta   │
│ next          │  │ final        │  │ = learning    │
│               │  │              │  │               │
│ the pushups.  │  │ the thesis.  │  │ the ruler.    │
└──────────────┘  └──────────────┘  └──────────────┘
```

If well drift decreases but judge scores don't improve → memorizing not learning.
If judge scores improve but well drift stays high → getting lucky not understanding.
Both signals should correlate. If they don't, something is wrong.

## File Structure

```
engram-harness/
│
├── src/main/java/engram/
│   │
│   │  the four nouns
│   ├── Brain.java              # .engram via Rust JNI
│   ├── Mouth.java              # local model interface (ollama)
│   ├── Hand.java               # tool: name + schema + do(args) → result
│   ├── Judge.java              # LLM + rubric card, certainty rounds
│   │
│   │  the runtime
│   ├── Session.java            # one conversation: brain+mouth+hands+history
│   ├── Harness.java            # loads config, wires everything
│   ├── Protocol.java           # JSON message format, transport-agnostic
│   │
│   │  transports (thin adapters)
│   ├── transport/
│   │   ├── Stdio.java          # JSON over stdin/stdout (pipe/LSP-style)
│   │   ├── Http.java           # REST + WebSocket (jdk.httpserver)
│   │   └── Repl.java           # interactive terminal (human-friendly)
│   │
│   │  executors (the three hands)
│   ├── executor/
│   │   ├── ProcessExecutor.java  # run_code, bash (ProcessBuilder)
│   │   ├── FileExecutor.java     # read_file, write_file (java.nio)
│   │   └── HttpExecutor.java     # web_search (HttpClient)
│   │
│   │  model adapters
│   ├── model/
│   │   ├── LocalModel.java     # mouth: ollama / llama.cpp / local only
│   │   └── RemoteModel.java    # judges only: any OpenAI-compatible API
│   │
│   │  entry point
│   └── Main.java               # CLI: chat, serve, measure, train, diff
│
├── src/main/resources/
│   └── tools/                  # tool definitions (JSON, data-driven)
│       ├── write_file.json
│       ├── read_file.json
│       ├── run_code.json
│       ├── web_search.json
│       └── bash.json
│
├── judges/                     # personality cards (JSON, hot-swappable)
│   ├── strict.json
│   ├── creative.json
│   ├── clarity.json
│   └── technical.json
│
├── native/                     # Rust JNI shared library (per-platform)
│   ├── libengram.so            # linux
│   ├── libengram.dylib         # mac
│   └── engram.dll              # windows
│
└── build.gradle
```

## Dependencies

Almost none.

```
Java 21+ stdlib only for core:
  • jdk.httpserver          HTTP transport (built-in since JDK 18)
  • java.net.http.HttpClient  LLM API calls (built-in since JDK 11)
  • java.nio                file I/O (built-in)
  • java.lang.ProcessBuilder  tool sandbox (built-in)
  • Virtual threads          concurrency (built-in since JDK 21)

One external dep:
  • com.google.code.gson     JSON (small, no deps of its own)

That's it. No Spring. No Netty. No Guice. No Lombok.
No dependency injection framework. No annotation processors.
Just Java that does what it says.
```

## Cross-Platform by Default

Not "cross-platform because we patched it." Cross-platform because we never assumed a platform.

- `java.nio.Path` everywhere. Never string paths. Never `/` vs `\`.
- `ProcessBuilder` for all process execution. JVM handles OS differences.
- `HttpClient` for all HTTP. JVM handles TLS, DNS, proxies.
- `System.getProperty("java.io.tmpdir")` for sandbox root.
- Rust native lib: compile per-platform, load at runtime. `System.loadLibrary("engram")` picks the right `.so`/`.dylib`/`.dll`.
- No shell assumptions: `run_code` uses `ProcessBuilder` directly, never `Runtime.exec("bash -c ...")`. That's the trap.
- `Paths.get(home, ".engram", "brains")` for brain storage.
- `StandardCharsets.UTF_8` everywhere. Never default charset.
- No `#ifdef`. No platform switch statements. Just Java.

Runs on: Linux, macOS, Windows, Android (via JNI).

## Future Layers

### Engram Diff

Compare two brain snapshots. Like `git diff` but for minds.

- Which wells grew (block count delta)
- Which wells are new (emerged since last snapshot)
- Inter-well distance shifts (topology changes)
- Dream buffer composition changes
- Use case: snapshot before/after a project, diff to see what she learned

### Engram Network

Decentralized well-delta sync over peer-to-peer mesh.

- Brains broadcast well deltas (new wells, significant centroid shifts)
- NOT full brain sync. Just well centroids + counts. Tiny payloads.
- No central server. Gossip protocol.
- A discovery by one brain propagates to all.
- Multiple Alexandrias learning from different humans, sharing what they find.

### Sleep / Consolidation

The trigger should come from the geometry, not arbitrary timers. Possible natural triggers:

- Dream buffer about to cycle out old entries
- High drift variance across recent conversations (confusion needs consolidating)
- Brain idle (no conversations, might as well dream)
- NOT "every N conversations" — that's fake biology

## Implementation Order

1. ~~Rust codec~~ (done — 73 tests, 4415 lines, all modules)
2. Genesis ingestion (running — Alexandria at 421K articles, phase 2 active)
3. **Rust JNI bridge** — expose Brain.measure/absorb/encode to Java via cdylib
4. **Brain.java + Harness.java** — load .engram, wire the four nouns
5. **Mouth.java + LocalModel.java** — ollama adapter, prompt building with well profile
6. **Session.java + Protocol.java** — the learning loop, JSON messages
7. **Repl.java** — terminal chat (the first working demo: `engram chat brain.engram`)
8. **Hand.java + executors** — tool definitions, sandbox, ProcessBuilder
9. **Stdio.java + Http.java** — pipe and HTTP transports
10. **Judge.java + RemoteModel.java** — scoring, certainty rounds, feedback absorption
11. **Task runner** — assignments, projects, checkpoints
12. **Diff** — brain snapshot comparison
13. **Network** — P2P well-delta gossip (someday)

Steps 3-7 get us to a working `engram chat` command. That's the MVP.
Steps 8-9 give her hands and make her accessible to any frontend.
Steps 10-11 let her learn from structured practice.
Steps 12-13 are the long game.
