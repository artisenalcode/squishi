# Graph Report - /home/alvin/Code/_labs/squishi  (2026-08-06)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 164 nodes · 286 edges · 16 communities (13 shown, 3 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 15

## God Nodes (most connected - your core abstractions)
1. `route()` - 21 edges
2. `compress_diff()` - 19 edges
3. `compress_log()` - 10 edges
4. `build_output()` - 9 edges
5. `DiffFile` - 8 edges
6. `run()` - 8 edges
7. `compress_json_array()` - 7 edges
8. `DiffHunk` - 6 edges
9. `compress_search_results()` - 6 edges
10. `SemanticDedup` - 6 edges

## Surprising Connections (you probably didn't know these)
- `route()` --calls--> `detect()`  [INFERRED]
  src/main.rs → src/content_detect.rs
- `route()` --calls--> `compress_diff()`  [INFERRED]
  src/main.rs → src/diff_compress.rs
- `route()` --calls--> `compress_json_array()`  [INFERRED]
  src/main.rs → src/json_compress.rs
- `route()` --calls--> `dedupe_line_runs()`  [INFERRED]
  src/main.rs → src/line_dedup.rs
- `route()` --calls--> `compress_log()`  [INFERRED]
  src/main.rs → src/log_compress.rs

## Import Cycles
- None detected.

## Communities (16 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (28): build_n_hunk_diff(), build_synthetic_diff(), combined_diff_3way_content_is_parsed_and_emitted(), compress_diff(), diff_combined_header_starts_a_file(), DiffCompressConfig, DiffCompressResult, DiffFile (+20 more)

### Community 1 - "Community 1"
Cohesion: 0.17
Nodes (23): ContentKind, Map, Option, Result, adversarial_content_survives_json_round_trip(), build_output(), Cli, diff_over_threshold_runs_diff_compressor() (+15 more)

### Community 2 - "Community 2"
Cohesion: 0.24
Nodes (14): cosine_similarity(), DedupResult, distinct_sentences_both_survive(), identical_sentence_repeated_collapses_to_one(), Result, Self, Session, String (+6 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (6): ContentKind, detect(), magika_label(), malformed_json_like_content_falls_through(), Option, String

### Community 4 - "Community 4"
Cohesion: 0.23
Nodes (15): caps_repeated_errors_to_max_errors_plus_first_last(), ClassifiedLine, classify(), compress_log(), context_lines_surround_selected_errors(), keeps_all_lines_when_under_budget(), keeps_summary_lines_regardless_of_error_cap(), LineLevel (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.27
Nodes (8): adversarial_content_round_trips_as_valid_json(), json_array_is_detected_and_compressed(), plain_prose_dedups_and_reports_char_counts(), Value, run(), run_json(), rust_source_is_classified_as_other(), short_diff_passes_through_with_expected_fields()

### Community 6 - "Community 6"
Cohesion: 0.29
Nodes (7): compress_json_array(), dedupes_exact_duplicate_elements(), JsonCompressResult, large_array_caps_to_first_and_last_edge(), Option, String, small_array_is_unchanged_besides_formatting()

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (3): dedupe_line_runs(), dedupe_never_lossily_truncates_non_repeating_content(), String

### Community 8 - "Community 8"
Cohesion: 0.48
Nodes (6): caps_many_matches_in_one_file(), compress_search_results(), few_matches_per_file_are_kept_as_is(), preserves_per_file_grouping_across_interleaved_input(), String, SearchCompressResult

### Community 9 - "Community 9"
Cohesion: 0.60
Nodes (4): classify(), main(), Result, Session

### Community 10 - "Community 10"
Cohesion: 0.40
Nodes (4): main(), Box, Error, Result

### Community 11 - "Community 11"
Cohesion: 0.40
Nodes (4): main(), Box, Error, Result

## Knowledge Gaps
- **2 isolated node(s):** `claude-code-posttooluse-hook.sh script`, `squishi`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `route()` connect `Community 1` to `Community 0`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`?**
  _High betweenness centrality (0.245) - this node is a cross-community bridge._
- **Why does `compress_diff()` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `compress_json_array()` connect `Community 6` to `Community 1`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `route()` (e.g. with `detect()` and `compress_diff()`) actually correct?**
  _`route()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `claude-code-posttooluse-hook.sh script`, `squishi` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.14705882352941177 - nodes in this community are weakly interconnected._