# Graph Report - squishi  (2026-08-19)

## Corpus Check
- 41 files · ~55,236 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 620 nodes · 1186 edges · 36 communities (30 shown, 6 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `befe06e8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- diff_compress.rs
- main.rs
- semantic_dedup.rs
- content_detect.rs
- log_compress.rs
- cli.rs
- json_compress.rs
- line_dedup.rs
- session_stats.rs
- toon.rs
- session_prune.rs
- invariants.rs
- pixel.rs
- What You Must Do When Invoked
- doctor.rs
- session_digest.rs
- punctuation_restore.rs
- squishi
- strip_read_tool_line_numbers
- Plan: port `session_to_trm.py` to Rust — squishi does extraction+digest+
- strip_base64_blobs
- graphify reference: extra exports and benchmark
- Plan: `session_prune` — structural pruning for squishi (Rust)
- Plan: deterministic base64-blob stripping (option A from 2026-08-07 /think)
- graphify reference: query, path, explain
- main
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- CLAUDE.md
- .claude/CLAUDE.md
- extraction-spec.md
- squishi

## God Nodes (most connected - your core abstractions)
1. `route()` - 21 edges
2. `compress_diff()` - 19 edges
3. `describe()` - 19 edges
4. `route_impl()` - 19 edges
5. `parse()` - 17 edges
6. `encode()` - 17 edges
7. `SemanticDedup` - 16 edges
8. `extract_session_text()` - 16 edges
9. `compress_json_array()` - 14 edges
10. `PunctuationRestorer` - 13 edges

## Surprising Connections (you probably didn't know these)
- `run_toon()` --references--> `Output`  [EXTRACTED]
  tests/cli.rs → src/main.rs
- `route_impl()` --calls--> `strip_base64_blobs()`  [INFERRED]
  src/main.rs → src/base64_strip.rs
- `route_impl()` --calls--> `detect()`  [INFERRED]
  src/main.rs → src/content_detect.rs
- `route_impl()` --calls--> `compress_diff()`  [INFERRED]
  src/main.rs → src/diff_compress.rs
- `route_impl()` --calls--> `compress_json_array()`  [INFERRED]
  src/main.rs → src/json_compress.rs

## Import Cycles
- None detected.

## Communities (36 total, 6 thin omitted)

### Community 0 - "diff_compress.rs"
Cohesion: 0.17
Nodes (28): build_n_hunk_diff(), build_synthetic_diff(), combined_diff_3way_content_is_parsed_and_emitted(), compress_diff(), diff_combined_header_starts_a_file(), DiffCompressConfig, DiffCompressResult, DiffFile (+20 more)

### Community 1 - "main.rs"
Cohesion: 0.10
Nodes (44): PathBuf, ContentKind, a_base64_blob_in_plain_text_is_reported_in_json_output(), a_base64_blob_inside_json_is_stripped_and_stays_valid_json(), adversarial_content_survives_json_round_trip(), build_output(), Cli, configs_for_level() (+36 more)

### Community 2 - "semantic_dedup.rs"
Cohesion: 0.06
Nodes (47): BertModel, Duration, main(), Box, Error, Result, PunctuationRestorer, Device (+39 more)

### Community 3 - "content_detect.rs"
Cohesion: 0.10
Nodes (13): copy_features(), detect(), extract_features(), is_whitespace(), magika_label(), malformed_json_like_content_falls_through(), Option, String (+5 more)

### Community 4 - "log_compress.rs"
Cohesion: 0.21
Nodes (16): caps_repeated_errors_to_max_errors_plus_first_last(), ClassifiedLine, classify(), compress_log(), context_lines_surround_selected_errors(), keeps_all_lines_when_under_budget(), keeps_summary_lines_regardless_of_error_cap(), LineLevel (+8 more)

### Community 5 - "cli.rs"
Cohesion: 0.11
Nodes (19): Path, adversarial_content_round_trips_as_valid_json(), json_array_is_detected_and_compressed(), json_semantic_dedup_exposes_full_kept_array_with_index_and_shape(), plain_prose_dedups_and_reports_char_counts(), Value, run(), run_json() (+11 more)

### Community 6 - "json_compress.rs"
Cohesion: 0.18
Nodes (20): RawValue, a_field_failing_the_safety_bar_is_withheld_even_when_another_field_is_disclosed(), build_marker(), compress_json_array(), dedupes_exact_duplicate_elements(), display_raw(), dropped_elements_marker_states_constant_enum_and_range_facts(), dropped_elements_with_a_unique_id_each_render_coverage_not_enumeration() (+12 more)

### Community 7 - "line_dedup.rs"
Cohesion: 0.29
Nodes (3): dedupe_line_runs(), dedupe_never_lossily_truncates_non_repeating_content(), String

### Community 8 - "session_stats.rs"
Cohesion: 0.15
Nodes (20): BTreeMap, caps_many_matches_in_one_file(), compress_search_results(), few_matches_per_file_are_kept_as_is(), preserves_per_file_grouping_across_interleaved_input(), String, SearchCompressResult, aggregates_real_squishi_calls_by_kind() (+12 more)

### Community 9 - "toon.rs"
Cohesion: 0.08
Nodes (42): Display, Formatter, Number, decode(), DecodeError, encode(), encode_array(), encode_bare_field_name() (+34 more)

### Community 10 - "session_prune.rs"
Cohesion: 0.16
Nodes (35): HashMap, apply_pruning(), apply_pruning_never_mutates_the_original_string(), apply_pruning_replaces_only_flagged_content_and_preserves_line_count(), collapse_task_launches(), collapse_task_launches_flags_all_but_the_latest_launch_notice(), collapse_task_launches_ignores_unrelated_content(), dedupe_latest_read() (+27 more)

### Community 11 - "invariants.rs"
Cohesion: 0.19
Nodes (23): a_credential_shaped_field_name_is_withheld_even_when_numeric(), a_credential_shaped_field_name_is_withheld_regardless_of_value_shape(), a_field_present_in_only_some_units_is_never_stated_as_a_bare_constant(), a_long_or_spaced_value_withholds_its_whole_field(), absent_bucket_is_accurate(), analyze_field(), constant_field_detected_and_rendered(), coverage_withholds_a_field_whose_extreme_values_are_long_or_spaced() (+15 more)

### Community 12 - "pixel.rs"
Cohesion: 0.13
Nodes (21): GrayImage, PSF2Font, blit_glyph(), content_longer_than_one_page_spills_onto_a_second_page(), dense_json_measures_well_under_the_profitability_threshold(), eligible_kind(), empty_text_still_renders_one_blank_page(), encode_png() (+13 more)

### Community 13 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 15 - "doctor.rs"
Cohesion: 0.14
Nodes (19): binary_identity_check_passes_and_reports_a_real_path(), Check, check_binary_identity(), check_hook_proxy_signal(), check_magika_loads(), check_model_cache_location(), check_semantic_dedup_loads(), DoctorReport (+11 more)

### Community 16 - "session_digest.rs"
Cohesion: 0.21
Nodes (20): a_second_call_with_the_first_calls_total_lines_yields_only_new_content(), build_digest_content(), build_digest_content_matches_the_expected_header_shape(), empty_transcript_produces_empty_text_and_zero_turns(), extract_session_text(), extracts_user_and_assistant_text_turns_in_order(), Option, String (+12 more)

### Community 17 - "punctuation_restore.rs"
Cohesion: 0.19
Nodes (13): capitalize_first(), long_input_is_chunked_and_fully_restored_without_dropping_words(), reconstruct(), reconstruct_handles_comma_without_capitalizing(), reconstruct_inserts_period_and_capitalizes_next_word(), reconstruct_question_mark_also_triggers_capitalization(), reconstruct_with_no_punctuation_labels_just_capitalizes_first_word(), restores_punctuation_on_real_unpunctuated_prose() (+5 more)

### Community 18 - "squishi"
Cohesion: 0.15
Nodes (12): Boundary, Detection: regex first, Magika as fallback — deliberately, not arbitrarily, Development, `diff_compress` — real port of headroom's `DiffCompressor`, CCR stripped, `--doctor` — self-diagnostics, `--level` — how hard each compressor pushes, `semantic_dedup` — wired in, replaces the earlier Kompress port, `session_digest` — extraction + compression for total-recall staging (+4 more)

### Community 19 - "strip_read_tool_line_numbers"
Cohesion: 0.38
Nodes (11): a_gap_in_the_sequence_leaves_content_completely_untouched(), a_real_numbered_tsv_id_column_that_happens_to_be_sequential_still_strips_safely(), content_with_no_numbering_is_returned_unchanged(), empty_content_is_not_trusted_as_real_read_tool_output(), numbered(), preserves_a_missing_trailing_newline(), real_captured_shape_with_an_embedded_fact_survives_intact(), String (+3 more)

### Community 20 - "Plan: port `session_to_trm.py` to Rust — squishi does extraction+digest+"
Cohesion: 0.18
Nodes (10): Context — ported logic, read in full from the real Python source, Goal, Non-goals, Part 1 — squishi: `src/session_digest.rs`, Part 2 — total-recall: `trm ingest-session <path>`, Plan: port `session_to_trm.py` to Rust — squishi does extraction+digest+, Real finding, checked before designing: `session_prune` doesn't, Risks (+2 more)

### Community 21 - "strip_base64_blobs"
Cohesion: 0.35
Nodes (10): a_base64_blob_inside_a_json_string_value_stays_valid_json(), a_real_data_uri_gets_replaced_with_one_marker(), a_real_jwt_payload_segment_is_not_mistaken_for_a_blob(), a_standalone_long_base64_run_gets_replaced(), content_with_no_base64_is_returned_unchanged(), marker(), multiple_blobs_in_one_document_are_all_counted(), String (+2 more)

### Community 22 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 23 - "Plan: `session_prune` — structural pruning for squishi (Rust)"
Cohesion: 0.25
Nodes (7): Context — real transcript shape, confirmed by reading this session's, Goal, Non-goals, Plan: `session_prune` — structural pruning for squishi (Rust), Risks, Steps, Validation

### Community 24 - "Plan: deterministic base64-blob stripping (option A from 2026-08-07 /think)"
Cohesion: 0.29
Nodes (6): Context, Goal, Plan: deterministic base64-blob stripping (option A from 2026-08-07 /think), Risks, Steps, Validation

### Community 25 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 26 - "main"
Cohesion: 0.40
Nodes (4): main(), Box, Error, Result

### Community 27 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 28 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 29 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

## Knowledge Gaps
- **73 isolated node(s):** `squishi`, `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+68 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `route_impl()` connect `main.rs` to `diff_compress.rs`, `semantic_dedup.rs`, `content_detect.rs`, `log_compress.rs`, `json_compress.rs`, `line_dedup.rs`, `session_stats.rs`, `strip_read_tool_line_numbers`, `strip_base64_blobs`?**
  _High betweenness centrality (0.148) - this node is a cross-community bridge._
- **Why does `SemanticDedup` connect `semantic_dedup.rs` to `main.rs`, `main`, `doctor.rs`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `Output` connect `main.rs` to `cli.rs`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `route_impl()` (e.g. with `strip_base64_blobs()` and `detect()`) actually correct?**
  _`route_impl()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `squishi`, `graphify`, `Usage` to the rest of the system?**
  _73 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `main.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.10434782608695652 - nodes in this community are weakly interconnected._
- **Should `semantic_dedup.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.06247086247086247 - nodes in this community are weakly interconnected._