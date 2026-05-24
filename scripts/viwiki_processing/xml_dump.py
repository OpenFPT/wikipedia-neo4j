import os
import sys
import time

import lxml.etree as ET


def get_namespace_prefix(root):
    tag = root.tag
    if isinstance(tag, ET.QName):
        tag_str = tag.text or ""
    elif isinstance(tag, (bytes, bytearray)):
        tag_str = tag.decode("utf-8", errors="ignore")
    else:
        tag_str = tag

    if tag_str.startswith("{"):
        return tag_str.split("}")[0] + "}"
    return ""


def page_tags(namespace_prefix):
    return {
        "page": f"{namespace_prefix}page",
        "title": f"{namespace_prefix}title",
        "ns": f"{namespace_prefix}ns",
        "id": f"{namespace_prefix}id",
        "revision": f"{namespace_prefix}revision",
        "text": f"{namespace_prefix}text",
    }


def revision_text(page_elem, tags):
    revision = page_elem.find(tags["revision"])
    if revision is None:
        return ""

    text_elem = revision.find(tags["text"])
    if text_elem is None:
        return ""

    return text_elem.text or ""


def clear_processed_element(elem):
    elem.clear()
    parent = elem.getparent()
    if parent is not None:
        while elem.getprevious() is not None:
            del parent[0]


def iter_pages(xml_path):
    context = ET.iterparse(xml_path, events=("start", "end"))
    context = iter(context)
    _, root = next(context)

    tags = page_tags(get_namespace_prefix(root))
    for event, elem in context:
        if event == "end" and elem.tag == tags["page"]:
            yield elem, tags


def parse_wiki_dump(xml_path, limit=None):
    """
    Streams a MediaWiki XML dump using lxml.etree.iterparse.
    Extremely fast and memory efficient by removing elements from the tree after
    processing them.
    """
    if not os.path.exists(xml_path):
        print(f"Error: File '{xml_path}' not found.", file=sys.stderr)
        return

    print(f"Starting to parse: {xml_path}")
    start_time = time.time()
    count = 0
    articles_count = 0

    for elem, tags in iter_pages(xml_path):
        count += 1

        title_elem = elem.find(tags["title"])
        ns_elem = elem.find(tags["ns"])
        id_elem = elem.find(tags["id"])

        title = title_elem.text if title_elem is not None else ""
        ns_val = ns_elem.text if ns_elem is not None else ""
        page_id = id_elem.text if id_elem is not None else ""

        if ns_val == "0":
            articles_count += 1
            yield {
                "id": page_id,
                "title": title,
                "text": revision_text(elem, tags),
            }

            if limit and articles_count >= limit:
                clear_processed_element(elem)
                break

        clear_processed_element(elem)

        if count % 10000 == 0:
            elapsed = time.time() - start_time
            print(
                f"Parsed {count} XML elements ({articles_count} articles) in {elapsed:.2f}s..."
            )

    elapsed = time.time() - start_time
    print(
        f"Finished parsing XML. Total elements: {count}, Articles (ns=0): {articles_count} (Took {elapsed:.2f}s)"
    )


def find_page_by_title(xml_path, target_title):
    """
    Streams the XML dump and searches for a page with the exact title
    (case-insensitive). Returns the raw wikitext and page details once found.
    """
    if not os.path.exists(xml_path):
        print(f"Error: File '{xml_path}' not found.", file=sys.stderr)
        return None

    print(f"Searching for '{target_title}' in {xml_path}...")
    target_lower = target_title.strip().lower()

    for elem, tags in iter_pages(xml_path):
        title_elem = elem.find(tags["title"])
        title = title_elem.text if title_elem is not None else ""

        if title.strip().lower() == target_lower:
            page_id = elem.find(tags["id"])
            page_id_val = page_id.text if page_id is not None else "Unknown"
            page = {
                "title": title,
                "id": page_id_val,
                "text": revision_text(elem, tags),
            }
            clear_processed_element(elem)
            return page

        clear_processed_element(elem)

    return None
