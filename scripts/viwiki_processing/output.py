def write_article_dump(title, page_id, url, text, heading, output_file="dump.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"ID: {page_id}\n")
        f.write(f"URL: {url}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{heading}\n")
        f.write("=" * 80 + "\n")
        f.write(text)
        f.write("\n")
