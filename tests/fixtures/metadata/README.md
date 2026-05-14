# Metadata fixtures

YAML examples of `eurpe.schema.ChunkMetadata` records — one per
`SourceStatus` value (`funded`, `rejected`, `esr_note`, `unknown`). They are
loaded by `tests/test_schema.py::test_chunk_metadata_round_trips_all_fixtures`
to prove that real-world-shaped records validate cleanly and survive a YAML
round trip with no data loss.
