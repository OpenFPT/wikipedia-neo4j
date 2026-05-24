# viwiki_processing

Tools for converting Vietnamese Wikipedia MediaWiki XML dumps to Parquet and
querying articles from either XML dumps or Parquet output.

## Commands

Install the optional XML/Parquet processing dependency group first:

```bash
uv sync --group xml-processing
```

Run the package CLI modules directly:

```bash
python -m scripts.viwiki_processing.cli convert --xml dumps/example.xml
python -m scripts.viwiki_processing.cli query-xml "Article title" --xml dumps/example.xml
python -m scripts.viwiki_processing.cli query-parquet "Article title" --parquet articles_cleaned
```

Conversion writes two Parquet dataset folders by default:

- `articles_cleaned/` — cleaned article text.
- `articles_raw/` — original raw wikitext for link/template graph extraction.

If `--xml` is omitted, conversion processes every `*.xml` file in `./dumps`.
Parquet folders are sharded as `part-00000.parquet`, `part-00001.parquet`, ...
with 100,000 articles per shard by default. Override that with `--batch-size`.

You can override either folder:

```bash
python -m scripts.viwiki_processing.cli convert \
  --xml dumps/example.xml \
  --output articles_cleaned \
  --raw-output articles_raw
```

The package also exposes console scripts when installed:

```bash
viwiki-convert --xml dumps/example.xml
viwiki-query-xml "Article title" --xml dumps/example.xml
viwiki-query-parquet "Article title" --parquet articles_cleaned
```

If `--xml` is omitted, XML commands look in `./dumps` and use the smallest dump
first. If `--parquet` is omitted, the Parquet query command tries
`articles_cleaned`, `viwiki_articles.parquet`, `output.parquet`, then
`test.parquet`.

## Structure

- `scripts/viwiki_processing/wikitext.py` — wikitext cleanup.
- `scripts/viwiki_processing/xml_dump.py` — streaming XML parsing and title lookup.
- `scripts/viwiki_processing/parquet.py` — XML-to-Parquet conversion, optional raw
  wikitext export, and Parquet lookup.
- `scripts/viwiki_processing/output.py` — shared `dump.txt` writing helpers.
- `scripts/viwiki_processing/cli.py` — command-line argument parsing.

Large dumps, generated Parquet files, and `dump.txt` are runtime artifacts rather
than source code.

## Wikitext parsing notes

The cleaner follows the same broad approach as Hugging Face's
`wikimedia/wikipedia` dataset script: parse with `mwparserfromhell`, remove
media/file links, remove `<ref>` and `<table>` tags, rewrite category links
without the category namespace prefix, then call `strip_code()` section by
section with explicit `normalize=True`, `collapse=True`, and
`keep_template_params=False`.

This is not a full MediaWiki renderer: templates are stripped rather than
expanded, so text produced by transclusion is not recoverable without a
template/rendering layer.

`clean_wikitext(..., skip_style_tags=True)` is available for malformed quote
markup, but the default keeps it disabled because enabling it can leave normal
bold/italic markers (`'''text'''`) in otherwise clean article text.
