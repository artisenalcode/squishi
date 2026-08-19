# squishi

Rust-native text compressor. Detects content shape (JSON, diff, log, search
results, base64 blob, plain prose, ...) and routes to the technique that
fits it, instead of one generic squeeze for everything.

Never ships a result that isn't actually smaller than the input.

squishi only compresses text it's handed — no store, no retrieve. That's
[total-recall](https://github.com/artisenalcode/total-recall)'s job.

## Install

Grab a prebuilt binary from the [releases page](https://github.com/artisenalcode/squishi/releases),
or build from source:

```sh
cargo build --release
```

`protobuf-compiler` is required on the build machine (used for the embedded
ONNX content-detection model).

## Usage

```sh
squishi "some text to compress"
echo "some text" | squishi
squishi --json "some text"          # full contract: compressed/kind/source/chars_before/chars_after/...
```

### Key flags

- `--level conservative|default|aggressive` — how hard each compressor pushes.
- `--force-kind plain-text` — skip shape detection, force a content kind.
- `--batch` — compress many texts in one process (reads a JSON array from
  stdin), avoiding a fresh model load per invocation.
- `--toon` — losslessly re-encode JSON as [TOON](https://github.com/toon-format/spec)
  instead of running the compression pipeline.
- `--doctor` — run self-diagnostics (`--quick` skips model-load checks).

### Claude Code session tooling

- `--session-prune <transcript.jsonl>` — structural pruning of a transcript
  (`--write <out>` to write a pruned copy, never mutates the input).
- `--session-digest <transcript.jsonl>` — extract + compress human/assistant
  prose into a stage-ready digest.
- `--session-stats <transcript.jsonl>` — report cumulative real savings from
  every squishi call already recorded in a transcript.

Run `squishi --help` for the full flag reference.

## Development

```sh
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

## License

MIT — see [LICENSE](LICENSE).
