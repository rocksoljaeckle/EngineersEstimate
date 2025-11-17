# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Engineers Estimate is a Streamlit web application that calculates engineer's cost estimates for construction projects by:
1. Extracting project items (item number, name, unit, quantity) from uploaded PDF tables
2. Matching those items against CDOT (Colorado Department of Transportation) historical cost data books
3. Retrieving Weighted Average Bids (WABs) for each item
4. Computing total project estimate with source citations

## Running the Application

```bash
streamlit run estimate_app.py
```

To expose the app via a public URL:
```bash
ssh -R 80:localhost:[PORT] localhost.run
```

The app is password-protected via `st.secrets.get("APP_PASSWORD", "change_me")`.

## Architecture

### Multi-Model Extraction Strategy
The app uses a **dual-extraction approach** with both OpenAI and Claude to maximize accuracy:
- **OpenAI**: Extracts project items using structured parsing with `responses.parse()` API
- **Claude**: Independently extracts the same items using document vision
- **Reconciliation**: `match_project_items()` in `estimate_utils.py` fuzzy-matches results from both models
  - Items where both agree → automatically accepted
  - Items with quantity discrepancies → re-extracted with `check_disputed_pair()`
  - Unmatched items → flagged for manual review in the UI

### Cost Data Matching Workflow
After extracting project items, the app searches historical cost books:
1. **Agent-based search**: Uses the `agents` library to create search agents with `ItemSearchWrapper` tools
2. **Multi-year fallback**: Searches cost books in descending chronological order (2024 → 2023 → 2022)
3. **WAB extraction**: Once matched, extracts the Weighted Average Bid using year-specific prompts
4. **Batch processing**: All items processed concurrently via `asyncio.gather()`

### Citation System
When using "unstract whisper" OCR mode:
- PDF and LLMWhisperer JSON response stored in session state
- `get_citation_images()` from `GlobalUtils.unstract_citation` generates highlighted PDF page images
- Citations use OpenAI to find relevant line numbers, then map to bounding boxes
- Results cached in `st.session_state['citation_cache']` by `{item_name}|{item_number}|{wab}` key
- Cache cleared on new estimate generation

### File Processing Modes
Three OCR options available via `st.pills`:
- **none**: Native PDF text extraction (fastest, least accurate for scanned docs)
- **google ocr**: Uses Google Cloud Vision API to add invisible text layer
- **unstract whisper**: LLMWhisperer API with line metadata (required for citations)

## Key Files

- **estimate_app.py**: Streamlit UI, file upload, results display, citation dialogs
- **estimate_utils.py**: Core logic for extraction, matching, WAB retrieval
- **config.toml**: API keys, model names, prompt file paths, year-specific WAB extraction prompts
- **prompts/**: Contains system prompts for extraction and search tasks
  - `openai_extract_project_items_prompt.md`: OpenAI extraction instructions
  - `claude_extract_project_items_prompt.md`: Claude extraction instructions
  - `search_extract_item_choice.md`: Agent instructions for cost data search
  - `extract_wab_prompts/extract_wab_prompt_{year}.md`: Year-specific WAB extraction

## Dependencies on Sibling Projects

This project imports from two sibling directories:
- **CDOTCostData**: Contains `ItemSearchWrapper` for searching cost data books, pickled cost items at `../cdotcostdata/config.toml`
- **GlobalUtils**: Contains OCR utilities (`whisper_pdf_text_extraction`, `google_ocr_pdf_text_overlay`) and citation generation (`get_citation_images`)

Config files from siblings are also loaded:
- `../cdotcostdata/config.toml`: Cost data book paths and years
- `../proposalwriting/ddl_config.toml`: OpenAI API key and file cache path

## Session State Management

Critical session state keys:
- `wab_items`: List of `(ProjectItem, year, wab)` tuples for successfully matched items
- `final_undecided_items`: Items that couldn't be confidently parsed or matched
- `citation_pdf_path`: Path to uploaded PDF (saved to `citation_pdfs/` directory)
- `unstract_response_json`: Full JSON from LLMWhisperer including line metadata
- `citation_cache`: Dict mapping citation keys to generated images
- `editor_key`: Counter incremented to force data editor refresh on changes

## API Usage Notes

### OpenAI
- Uses `responses.parse()` with `text_format` for structured extraction (requires gpt-5 or compatible)
- Files cached by SHA256 hash in `openai_files_cache_path` to avoid re-uploads
- Uses `AsyncOpenAI` for concurrent requests

### Anthropic Claude
- Uses document vision with base64-encoded PDFs
- Prefills assistant message with `'['` or `'{'` to force JSON array/object output
- Uses `AsyncAnthropic` for concurrent requests

### LLMWhisperer (Unstract)
- Returns `result_text` (extracted text) and `line_metadata` (bounding boxes)
- Line metadata format: `[page_num, y_bottom, height, page_height]` per line
- API has rate limiting; retries with exponential backoff implemented

## Data Models (Pydantic)

```python
ProjectItem(item_number: str, item_name: str, unit: str, quantity: int|float)
ProjectItemsList(items: list[ProjectItem])
ItemChoice(item_id: int, item_number: str, item_name: str)
WeightedAverageBid(weighted_average_average_bid_for_year: float|None)
```

## Concurrency Patterns

All async operations use `asyncio.gather()` for parallelization:
- Dual model extraction (OpenAI + Claude) runs concurrently
- Disputed item re-checking runs in parallel for all disputed pairs
- WAB retrieval for all matched items runs concurrently
- Each item searches cost books sequentially (2024 → 2023 → 2022) but items are processed in parallel

## Citation PDF Storage

PDFs uploaded for citation are saved persistently to `citation_pdfs/` directory (not temp files) because:
- Citations may be requested multiple times during session
- TemporaryDirectory cleanup would delete files needed for later citations
- Directory is created automatically if it doesn't exist
