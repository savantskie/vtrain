"""
Extract clean text from a Wikipedia XML dump.
Strips markup, templates, tables, and other noise.
Writes output as a single .txt file, one article per line.
"""

import bz2
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def strip_markup(text: str) -> str:
    # Remove templates {{...}}
    while '{{' in text:
        start = text.find('{{')
        depth = 0
        for i in range(start, len(text)):
            if text[i:i+2] == '{{':
                depth += 1
            elif text[i:i+2] == '}}':
                depth -= 1
                if depth == 0:
                    text = text[:start] + text[i+2:]
                    break
        else:
            break

    # Remove tables {|...|}
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)

    # Remove file/image links
    text = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)

    # Convert [[link|display]] → display, [[link]] → link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

    # Remove external links [url text] → text
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove section headers === ... ===
    text = re.sub(r'={2,}[^=]+=+', '', text)

    # Remove bold/italic markup
    text = re.sub(r"'{2,}", '', text)

    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def extract(dump_path: str, output_path: str, max_articles: int = None):
    dump_path   = Path(dump_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ns = 'http://www.mediawiki.org/xml/DTD/MediaWiki'

    print(f"Extracting from {dump_path.name}...")
    count = 0

    with bz2.open(dump_path, 'rt', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        in_text    = False
        cur_title  = ''
        cur_ns     = ''
        buffer     = []

        for event, elem in ET.iterparse(f_in, events=('start', 'end')):
            tag = elem.tag.split('}')[-1]  # strip namespace

            if event == 'end' and tag == 'title':
                cur_title = elem.text or ''

            if event == 'end' and tag == 'ns':
                cur_ns = elem.text or ''

            if event == 'end' and tag == 'text':
                # ns=0 is article namespace — skip talk pages, user pages etc
                if cur_ns == '0' and elem.text:
                    text = strip_markup(elem.text)
                    # Skip stubs and very short articles
                    if len(text) > 200:
                        line = text.replace('\n', ' ').strip()
                        f_out.write(line + '\n')
                        count += 1

                        if count % 1000 == 0:
                            print(f"  {count} articles extracted...")

                        if max_articles and count >= max_articles:
                            break

                elem.clear()

    print(f"Done — {count} articles written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract clean text from a Wikipedia XML dump")
    parser.add_argument("--dump",    required=True, help="Path to .xml.bz2 Wikipedia dump file")
    parser.add_argument("--output",  required=True, help="Path to write extracted text file")
    parser.add_argument("--max-articles", type=int, default=None, help="Stop after N articles (default: all)")
    args = parser.parse_args()

    extract(args.dump, args.output, args.max_articles)
