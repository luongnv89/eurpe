# PDF fixtures

This directory is intentionally empty in version control.

* Real proposal PDFs are confidential and **must never be committed**. The
  repository-level `.gitignore` excludes `*.pdf` repo-wide for exactly this
  reason — see `.gitignore` at the project root.
* The Docling integration test
  (`tests/test_docling_parser.py::test_parse_real_pdf_returns_parsed_proposal`,
  marker `@pytest.mark.docling`) generates a tiny synthetic one-page PDF at
  test time using `reportlab` into a `tmp_path` and tears it down
  automatically — no on-disk fixture is needed.
* If you want to manually parse a local proposal PDF for spot-checking, drop
  it into this directory and run `eurpe ingest path/to/file.pdf`. The
  `.gitignore` rule will keep it out of git.
