import argparse
import os
import sys

from scripts.viwiki_processing.output import write_article_dump

DEFAULT_CLEAN_OUTPUT_DIR = "articles_cleaned"
DEFAULT_RAW_OUTPUT_DIR = "articles_raw"
DEFAULT_SHARD_SIZE = 100000
XML_PROCESSING_DEPS = {"lxml", "mwparserfromhell", "pyarrow"}


def require_xml_processing_dependency(error):
    if error.name not in XML_PROCESSING_DEPS:
        raise error

    print(
        "Error: XML processing dependencies are not installed. "
        "Run `uv sync --group xml-processing` first.",
        file=sys.stderr,
    )
    sys.exit(1)


def load_parquet_module():
    try:
        from scripts.viwiki_processing import parquet
    except ModuleNotFoundError as error:
        require_xml_processing_dependency(error)
        raise AssertionError("unreachable") from error
    return parquet


def load_wikitext_module():
    try:
        from scripts.viwiki_processing import wikitext
    except ModuleNotFoundError as error:
        require_xml_processing_dependency(error)
        raise AssertionError("unreachable") from error
    return wikitext


def load_xml_dump_module():
    try:
        from scripts.viwiki_processing import xml_dump
    except ModuleNotFoundError as error:
        require_xml_processing_dependency(error)
        raise AssertionError("unreachable") from error
    return xml_dump


def find_xml_dumps(dumps_dir="./dumps"):
    if not os.path.exists(dumps_dir):
        return []

    xml_files = [f for f in os.listdir(dumps_dir) if f.endswith(".xml")]
    xml_files.sort(key=lambda x: os.path.getsize(os.path.join(dumps_dir, x)))
    return [os.path.join(dumps_dir, f) for f in xml_files]


def build_convert_parser():
    parser = argparse.ArgumentParser(
        description="Convert Wikipedia MediaWiki XML dumps to Parquet format matching a reference schema."
    )
    add_convert_args(parser)
    return parser


def build_query_xml_parser():
    parser = argparse.ArgumentParser(
        description="Query Wikipedia XML dump by article title and dump raw wikitext to dump.txt."
    )
    add_query_xml_args(parser)
    return parser


def build_query_parquet_parser():
    parser = argparse.ArgumentParser(
        description="Query Wikipedia Parquet dataset by article title and write wikitext/plain text to dump.txt."
    )
    add_query_parquet_args(parser)
    return parser


def add_convert_args(parser):
    parser.add_argument(
        "--xml",
        type=str,
        nargs="+",
        help="One or more input MediaWiki XML files. If omitted, all ./dumps/*.xml files are converted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_CLEAN_OUTPUT_DIR,
        help="Path to cleaned Parquet dataset folder or .parquet file.",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=DEFAULT_RAW_OUTPUT_DIR,
        help="Path to raw-wikitext Parquet dataset folder or .parquet file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Articles per Parquet shard/row group.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of articles to process.",
    )


def add_query_xml_args(parser):
    parser.add_argument("title", type=str, nargs="?", help="The title of the Wikipedia article.")
    parser.add_argument("--xml", type=str, help="Specific XML dump file to search.")


def add_query_parquet_args(parser):
    parser.add_argument("title", type=str, nargs="?", help="The title of the Wikipedia article.")
    parser.add_argument("--parquet", type=str, help="Path to Parquet file to query.")


def convert_main(argv=None):
    args = build_convert_parser().parse_args(argv)
    run_convert(args)


def query_xml_main(argv=None):
    args = build_query_xml_parser().parse_args(argv)
    run_query_xml(args)


def query_parquet_main(argv=None):
    args = build_query_parquet_parser().parse_args(argv)
    run_query_parquet(args)


def run_convert(args):
    parquet = load_parquet_module()
    xml_paths = args.xml
    if not xml_paths:
        xml_paths = find_xml_dumps()
        if not xml_paths:
            print(
                "Error: No XML files found in ./dumps directory. Please specify --xml.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"No --xml provided. Converting all XML dumps: {', '.join(xml_paths)}")

    parquet.convert_xml_dumps_to_parquet(
        xml_paths=xml_paths,
        parquet_path=args.output,
        batch_size=args.batch_size,
        limit=args.limit,
        raw_parquet_path=args.raw_output,
    )


def run_query_xml(args):
    parquet = load_parquet_module()
    wikitext = load_wikitext_module()
    xml_dump = load_xml_dump_module()
    target_title = args.title
    if not target_title:
        target_title = input("Enter Wikipedia article title to dump: ").strip()
        if not target_title:
            print("Error: No title provided.")
            sys.exit(1)

    if args.xml:
        xml_paths = [args.xml]
    else:
        xml_paths = find_xml_dumps()
        if not xml_paths:
            print("Error: No XML dumps found in ./dumps directory.", file=sys.stderr)
            sys.exit(1)

    page_data = None
    for path in xml_paths:
        page_data = xml_dump.find_page_by_title(path, target_title)
        if page_data:
            break

    if not page_data:
        print(f"\n✗ Article with title '{target_title}' was not found in any XML dump.")
        sys.exit(1)

    output_file = "dump.txt"
    try:
        cleaned_text = wikitext.clean_wikitext(page_data["text"])
        write_article_dump(
            title=page_data["title"],
            page_id=page_data["id"],
            url=parquet.article_url(page_data["title"]),
            text=cleaned_text,
            heading="CLEANED PLAIN TEXT CONTENT:",
            output_file=output_file,
        )
        print(f"\n✓ Success! Found page '{page_data['title']}' (ID: {page_data['id']}).")
        print(f"Cleaned and written plaintext dump to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def run_query_parquet(args):
    parquet = load_parquet_module()
    target_title = args.title
    if not target_title:
        target_title = input("Enter Wikipedia article title to query from Parquet: ").strip()
        if not target_title:
            print("Error: No title provided.")
            sys.exit(1)

    parquet_path = args.parquet
    if not parquet_path:
        possible_paths = [
            DEFAULT_CLEAN_OUTPUT_DIR,
            "viwiki_articles.parquet",
            "output.parquet",
            "test.parquet",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                parquet_path = path
                break

        if not parquet_path:
            print(
                "Error: No Parquet files found in directory. Please specify --parquet.",
                file=sys.stderr,
            )
            sys.exit(1)

    page_data = parquet.query_article_from_parquet(parquet_path, target_title)
    if not page_data:
        print(f"✗ Article '{target_title}' not found in Parquet dataset.")
        sys.exit(1)

    output_file = "dump.txt"
    try:
        write_article_dump(
            title=page_data["title"],
            page_id=page_data["id"],
            url=page_data["url"],
            text=page_data["text"],
            heading="PLAIN TEXT CONTENT:",
            output_file=output_file,
        )
        print(f"✓ Success! Written plain text dump to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vietnamese Wikipedia dump processing tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a MediaWiki XML dump to Parquet.",
        description="Convert Wikipedia MediaWiki XML dumps to Parquet format matching a reference schema.",
    )
    add_convert_args(convert_parser)
    convert_parser.set_defaults(func=run_convert)

    query_xml_parser = subparsers.add_parser(
        "query-xml",
        help="Query an XML dump by article title.",
        description="Query Wikipedia XML dump by article title and dump raw wikitext to dump.txt.",
    )
    add_query_xml_args(query_xml_parser)
    query_xml_parser.set_defaults(func=run_query_xml)

    query_parquet_parser = subparsers.add_parser(
        "query-parquet",
        help="Query a Parquet dataset by article title.",
        description="Query Wikipedia Parquet dataset by article title and write wikitext/plain text to dump.txt.",
    )
    add_query_parquet_args(query_parquet_parser)
    query_parquet_parser.set_defaults(func=run_query_parquet)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
