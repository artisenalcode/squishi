# Graph Report - /home/alvin/Code/_labs/squishi  (2026-08-06)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 130 nodes · 224 edges · 12 communities (11 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.8)
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

## God Nodes (most connected - your core abstractions)
1. `compress_diff()` - 19 edges
2. `compress_log()` - 10 edges
3. `DiffFile` - 8 edges
4. `compress_json_array()` - 7 edges
5. `main()` - 7 edges
6. `DiffHunk` - 6 edges
7. `compress_search_results()` - 6 edges
8. `SemanticDedup` - 6 edges
9. `detect()` - 5 edges
10. `ParsedDiff` - 5 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `compress_diff()`  [INFERRED]
  src/main.rs → src/diff_compress.rs
- `main()` --calls--> `compress_json_array()`  [INFERRED]
  src/main.rs → src/json_compress.rs
- `main()` --calls--> `dedupe_line_runs()`  [INFERRED]
  src/main.rs → src/line_dedup.rs
- `main()` --calls--> `compress_log()`  [INFERRED]
  src/main.rs → src/log_compress.rs
- `main()` --calls--> `compress_search_results()`  [INFERRED]
  src/main.rs → src/search_compress.rs

## Import Cycles
- None detected.

## Communities (12 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (28): build_n_hunk_diff(), build_synthetic_diff(), combined_diff_3way_content_is_parsed_and_emitted(), compress_diff(), diff_combined_header_starts_a_file(), DiffCompressConfig, DiffCompressResult, DiffFile (+20 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (10): ContentKind, detect(), magika_label(), malformed_json_like_content_falls_through(), Option, String, Cli, main() (+2 more)

### Community 2 - "Community 2"
Cohesion: 0.24
Nodes (14): cosine_similarity(), DedupResult, distinct_sentences_both_survive(), identical_sentence_repeated_collapses_to_one(), Result, Self, Session, String (+6 more)

### Community 3 - "Community 3"
Cohesion: 0.23
Nodes (15): caps_repeated_errors_to_max_errors_plus_first_last(), ClassifiedLine, classify(), compress_log(), context_lines_surround_selected_errors(), keeps_all_lines_when_under_budget(), keeps_summary_lines_regardless_of_error_cap(), LineLevel (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.29
Nodes (7): compress_json_array(), dedupes_exact_duplicate_elements(), JsonCompressResult, large_array_caps_to_first_and_last_edge(), Option, String, small_array_is_unchanged_besides_formatting()

### Community 5 - "Community 5"
Cohesion: 0.29
Nodes (3): dedupe_line_runs(), dedupe_never_lossily_truncates_non_repeating_content(), String

### Community 6 - "Community 6"
Cohesion: 0.48
Nodes (6): caps_many_matches_in_one_file(), compress_search_results(), few_matches_per_file_are_kept_as_is(), preserves_per_file_grouping_across_interleaved_input(), String, SearchCompressResult

### Community 7 - "Community 7"
Cohesion: 0.60
Nodes (4): classify(), main(), Result, Session

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): main(), Box, Error, Result

### Community 9 - "Community 9"
Cohesion: 0.40
Nodes (4): main(), Box, Error, Result

## Knowledge Gaps
- **1 isolated node(s):** `squishi`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 1` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.277) - this node is a cross-community bridge._
- **Why does `compress_diff()` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `compress_json_array()` connect `Community 4` to `Community 1`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **What connects `squishi` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.11688311688311688 - nodes in this community are weakly interconnected._