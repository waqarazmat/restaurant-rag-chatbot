import json
import os
import time
import deepl
from firecrawl import Firecrawl
from dotenv import load_dotenv

load_dotenv()

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "fc-2e67d7cd0e1c4d16a5112a7d8713ebcf")
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY")

if not DEEPL_API_KEY:
    raise ValueError("❌ Please set DEEPL_API_KEY in your .env file!")

firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)
translator = deepl.Translator(DEEPL_API_KEY)


def translate_to_english(text: str) -> str:
    """Translate text to English using DeepL. Returns original text on failure."""
    if not text or not text.strip():
        return text
    try:
        result = translator.translate_text(text, target_lang="EN-US")
        return result.text
    except Exception as e:
        print(f"  ⚠️ Translation failed: {e} — keeping original text")
        return text


# ── 1. Scrape & translate the PDF menu ───────────────────────────────────────
pdf_url = "https://drugstorehasselt.be/wp-content/uploads/2022/12/2022_menu_binnenwerk_aanpassingen_v07_web.pdf"

print("📄 Scraping PDF Menu...")
try:
    scrape_result = firecrawl.scrape(pdf_url, formats=["markdown"])
    dutch_markdown = scrape_result.markdown if hasattr(scrape_result, "markdown") else ""

    if not dutch_markdown:
        print("❌ Nothing scraped from the PDF — it may be empty or inaccessible.")
        exit()

    print("✅ PDF scraped successfully!")
    print("🔄 Translating PDF menu to English with DeepL...")
    english_markdown = translate_to_english(dutch_markdown)
    print("✅ Translation complete!")

    pdf_document = {"url": pdf_url, "markdown": english_markdown}
    with open("pdf_menu_translated.json", "w", encoding="utf-8") as f:
        json.dump([pdf_document], f, indent=4, ensure_ascii=False)

    print("💾 Saved to pdf_menu_translated.json\n")

except Exception as e:
    print(f"❌ PDF scrape failed: {e}")


# ── 2. Scrape & translate the website pages ───────────────────────────────────
print("🗺️  Mapping website pages...")
try:
    map_result = firecrawl.map(url="https://drugstorehasselt.be/", limit=15)

    raw_urls = []
    if map_result.links:
        for item in map_result.links:
            url_str = str(item.url) if hasattr(item, "url") else str(item)
            raw_urls.append(url_str)

    print(f"Found {len(raw_urls)} pages. Starting scraping & translation...\n")

    all_translated_pages = []

    for url in raw_urls:
        print(f"Processing: {url}")
        try:
            scrape_result = firecrawl.scrape(url, formats=["markdown"])
            dutch_markdown = scrape_result.markdown if hasattr(scrape_result, "markdown") else ""

            if not dutch_markdown:
                print("  ⚠️ Empty page — skipping")
                continue

            if "Hello world" in dutch_markdown and "WordPress" in dutch_markdown:
                print("  ⏭️ Default WordPress page — skipping")
                continue

            print("  🔄 Translating with DeepL...")
            english_markdown = translate_to_english(dutch_markdown)

            all_translated_pages.append({"url": url, "markdown": english_markdown})
            print("  ✅ Done!")

            time.sleep(1)  # Be polite to the server

        except Exception as e:
            print(f"  ❌ Error on {url}: {e}")

    with open("website_data.json", "w", encoding="utf-8") as f:
        json.dump(all_translated_pages, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 Done! {len(all_translated_pages)} pages saved to website_data.json")

except Exception as e:
    print(f"❌ Website scrape failed: {e}")
