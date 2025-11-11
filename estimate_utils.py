import agents
from agents import Agent, Runner, function_tool, set_default_openai_key, RunConfig, trace
import re
import tomli
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel
from pathlib import Path
import json
import hashlib
from rapidfuzz import process, fuzz
import asyncio
import sys
from rapidfuzz import fuzz
import base64
import warnings
from functools import partial


from CDOTCostData.cost_data_utils import ItemSearchWrapper

# <editor-fold desc="openai file uploading & caching">
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(cache_path) -> dict:
    cache_path = cache_path
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_cache(cache_path: Path, cache: dict):
    cache_path.write_text(json.dumps(cache))


async def get_or_upload_async(file_path: str, client: AsyncOpenAI, cache_path: str, purpose="user_data") -> str:
    cache_path = Path(cache_path)
    file_path = Path(file_path)
    cache = load_cache(cache_path)
    digest = sha256(file_path)

    # 1. Cache hit ➜ just return the ID
    if digest in cache:
        return cache[digest]

    # 2. Cache miss ➜ upload once
    with open(file_path, "rb") as f:
        creation_resp = await client.files.create(file=f, purpose=purpose)
        file_id = creation_resp.id

    # 3. Remember it for next time
    cache[digest] = file_id
    save_cache(cache_path, cache)
    return file_id
# </editor-fold>

class ProjectItem(BaseModel):
    item_number: str
    item_name: str
    unit: str
    quantity: int | float

class ProjectItemsList(BaseModel):
    items: list[ProjectItem]

class ItemChoice(BaseModel):
    item_id: int
    item_number: str
    item_name: str

class WeightedAverageBid(BaseModel):
    weighted_average_average_bid_for_year: float | None


@function_tool
def report_item_choice(item_choice: ItemChoice) -> str:
    '''Report the item result corresponding to the weighted average bid for the year.'''
    return item_choice

@function_tool
def report_failure_to_find_item(explanation: str) -> str:
    '''Report failure to find item in cost data.'''
    return explanation

def confirm_items_match(item1: ProjectItem, item2: dict, item_name_match_threshold: float = 80.) -> bool:
    '''Raise an error if item1 (ProjectItem) and item2 (dict) do not match in name and unit.'''
    if item1.item_number.strip().lower() != item2['item_number'].strip().lower():
        return False
    if item1.unit.strip().lower() != item2['unit'].strip().lower():
        return False
    if fuzz.ratio(item1.item_name.lower().strip(), item2['item_name'].lower().strip()) < item_name_match_threshold:
        return False
    return True

def match_project_items(openai_project_items_list: ProjectItemsList, claude_project_items_list: list[dict], item_name_match_threshold: float = 80.):
    '''Compare two lists of project items extracted by different models, and return a tuple of (matched_items, disputed_items, openai_unmatched_items, claude_unmatched_items)'''
    scores = []
    for openai_idx, openai_item in enumerate(openai_project_items_list):
        for claude_idx, claude_item in enumerate(claude_project_items_list):
            score = fuzz.ratio(openai_item.item_number.lower().strip(), claude_item['item_number'].lower().strip())
            score += fuzz.ratio(openai_item.item_name.lower().strip(), claude_item['item_name'].lower().strip())
            score += fuzz.ratio(openai_item.unit.lower().strip(), claude_item['unit'].lower().strip())
            scores.append((score, openai_idx, claude_idx))
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    matched_openai_indices = set()
    matched_claude_indices = set()
    matched_items = []
    disputed_items = [] # list of pairs of [openai_item, claude_item] that have been matched, but have quantity discrepancies
    for score, openai_idx, claude_idx in scores:
        if openai_idx in matched_openai_indices or claude_idx in matched_claude_indices:
            continue
        openai_item = openai_project_items_list[openai_idx]
        claude_item = claude_project_items_list[claude_idx]
        if confirm_items_match(openai_item, claude_item, item_name_match_threshold):
            if abs(openai_item.quantity - claude_item['quantity']) > 0.1:
                disputed_items.append([openai_item, claude_item])
            else:
                matched_items.append(openai_item)
            matched_openai_indices.add(openai_idx)
            matched_claude_indices.add(claude_idx)
    openai_unmatched_items = [openai_project_items_list[i] for i in range(len(openai_project_items_list)) if i not in matched_openai_indices]
    claude_unmatched_items = [claude_project_items_list[i] for i in range(len(claude_project_items_list)) if i not in matched_claude_indices]
    return matched_items, disputed_items, openai_unmatched_items, claude_unmatched_items

async def check_disputed_pair(
        openai_item: ProjectItem,
        claude_item: dict,
        project_items_table_path: str,
        openai_async_client: AsyncOpenAI, openai_files_cache_path: str, openai_model: str,
        async_claude_client: AsyncAnthropic, claude_model: str,
        item_name_match_threshold: float = 80.,
        project_items_table_ocr_str: str|None=None
):
    item_name = None
    if fuzz.ratio(openai_item.item_name.lower().strip(),
                  claude_item['item_name'].lower().strip()) >= item_name_match_threshold:
        item_name = openai_item.item_name
    item_number = None
    if openai_item.item_number == claude_item['item_number']:
        item_number = openai_item.item_number
    unit = None
    if openai_item.unit == claude_item['unit']:
        unit = openai_item.unit
    extraction_prompt = 'Please extract the item number, item name, unit, and quantity for the item described below, from the attached table: \n'
    if item_number is not None:
        extraction_prompt += f'Item Number: {item_number}\n'
    if item_name is not None:
        extraction_prompt += f'Item Name: {item_name}\n'
    if unit is not None:
        extraction_prompt += f'Unit: {unit}\n'
    extraction_prompt += 'Respond with a JSON object with the following fields: item_number (string), item_name (string), unit (string), quantity (number). If any field is missing or cannot be determined, set it to null. Do not include any other text.'

    # claude extraction
    claude_check_inputs = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': extraction_prompt
                },
                {
                    'type': 'document',
                    'source': {
                        'type': 'base64',
                        'media_type': 'application/pdf',
                        'data': base64_pdf_string
                    }
                }
            ]
        },
        {
            'role': 'assistant',
            'content': '{'
        }
    ]
    claude_check_couroutine = async_claude_client.messages.create(model=claude_model,
                                                                  messages=claude_check_inputs,
                                                                  max_tokens=10_000)

    # openai extraction
    openai_check_inputs = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'input_text',
                    'text': extraction_prompt
                },
                {
                    'type': 'input_file',
                    'file_id': project_items_file_id
                }]
        }
    ]
    openai_check_coroutine = openai_async_client.responses.parse(
        model=openai_model,
        input=openai_check_inputs,
        text_format=ProjectItem,
        metadata={'source': 'ee_project_items_checking'}
    )
    check_results = await asyncio.gather(openai_check_coroutine, claude_check_couroutine)
    openai_check_item = check_results[0].output_parsed
    claude_check_item = json.loads('{'+check_results[1].content[0].text)
    if (
            openai_check_item.item_number == claude_check_item['item_number']
            and fuzz.ratio(openai_check_item.item_name.strip().lower(), claude_check_item['item_name'].strip().lower()) >= item_name_match_threshold
            and openai_check_item.unit.strip().lower() == claude_check_item['unit'].strip().lower()
            and abs(openai_check_item.quantity - claude_check_item['quantity']) < 0.1
    ):
        return True, openai_check_item
    else:
        final_item = {}
        if openai_check_item.item_number == claude_check_item['item_number']:
            final_item['item_number'] = openai_check_item.item_number
        if fuzz.ratio(openai_check_item.item_name.strip().lower(), claude_check_item['item_name'].strip().lower()) >= item_name_match_threshold:
            final_item['item_name'] = openai_check_item.item_name
        if openai_check_item.unit.strip().lower() == claude_check_item['unit'].strip().lower():
            final_item['unit'] = openai_check_item.unit
        if abs(openai_check_item.quantity - claude_check_item['quantity']) < 0.1:
            final_item['quantity'] = openai_check_item.quantity
        return False, final_item

async def get_project_items(
        project_items_table_path: str,
        openai_async_client: AsyncOpenAI, openai_files_cache_path: str, openai_extract_project_items_prompt: str, openai_model: str,
        async_claude_client: AsyncAnthropic, claude_extract_project_items_prompt: str, claude_model: str,
        item_name_match_threshold: float = 80.,
        project_items_table_ocr_str: str|None=None
):
    assert project_items_table_path.endswith(".pdf")
    #openai extraction
    project_items_file_id = await get_or_upload_async(project_items_table_path, openai_async_client, openai_files_cache_path, purpose="user_data")
    openai_extraction_inputs = [
        {
            'role': 'system',
            'content': openai_extract_project_items_prompt
        },
        {
            'role': 'user',
            'content': [{
                'type': 'input_file',
                'file_id': project_items_file_id
            }]
        }
    ]
    if project_items_table_ocr_str is not None:
        openai_extraction_inputs[1]['content'].append({
            'type': 'input_text',
            'text': f'The following is the OCR text extracted from the PDF, which you should cross-reference with the table: \n{project_items_table_ocr_str}'
        })
    openai_extraction_coroutine = openai_async_client.responses.parse(
        model=openai_model,
        input=openai_extraction_inputs,
        text_format=ProjectItemsList,
        metadata={'source': 'ee_project_items_extraction'}
    )

    # claude extraction
    with open(project_items_table_path, 'rb') as f:
        pdf_data = f.read()
    base64_pdf_data = base64.standard_b64encode(pdf_data)
    base64_pdf_string = base64_pdf_data.decode('utf-8')
    claude_extraction_inputs = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': claude_extract_project_items_prompt
                },
                {
                    'type': 'document',
                    'source': {
                        'type': 'base64',
                        'media_type': 'application/pdf',
                        'data': base64_pdf_string
                    }
                }
            ]
        },
        {
            'role': 'assistant',
            'content': '['
        }
    ]
    if project_items_table_ocr_str is not None:
        claude_extraction_inputs[0]['content'].append({
            'type': 'text',
            'text': f'The following is the OCR text extracted from the PDF, which you should cross-reference with the table: \n{project_items_table_ocr_str}'
        })
    claude_extraction_couroutine = async_claude_client.messages.create(model=claude_model, messages=claude_extraction_inputs, max_tokens=10_000)

    openai_extraction_response, claude_extraction_response = await asyncio.gather(openai_extraction_coroutine, claude_extraction_couroutine)
    openai_project_items_list = openai_extraction_response.output_parsed.items
    claude_project_items_list = json.loads('['+claude_extraction_response.content[0].text)

    matched_items, disputed_items, openai_unmatched_items, claude_unmatched_items = match_project_items(openai_project_items_list, claude_project_items_list, item_name_match_threshold=item_name_match_threshold)

    final_undecided_items = claude_unmatched_items
    final_undecided_items.extend([item.model_dump() for item in openai_unmatched_items])

    check_disputed_pair_partial = partial(
        check_disputed_pair,
        project_items_table_path=project_items_table_path,
        openai_async_client=openai_async_client, openai_files_cache_path=openai_files_cache_path, openai_model=openai_model,
        async_claude_client=async_claude_client, claude_model=claude_model,
        item_name_match_threshold=item_name_match_threshold,
        project_items_table_ocr_str=project_items_table_ocr_str
    )
    checking_coroutine_pairs = [
        check_disputed_pair_partial(openai_item = openai_item, claude_item = claude_item)
        for openai_item, claude_item in disputed_items
    ]
    disputed_results = await asyncio.gather(*checking_coroutine_pairs)

    for is_match, final_item in disputed_results:
        if is_match:
            matched_items.append(final_item)
        else:
            final_undecided_items.append(final_item)
    return matched_items, final_undecided_items


async def get_item_weighted_avg_bid(
        project_item: ProjectItem,
        item_search_wrappers: list[ItemSearchWrapper], # in descending chronological order, most recent first
        search_agent_prompt: str, extract_wab_prompts: dict[int, str],
        async_client: AsyncOpenAI, model: str = 'gpt-5'
) -> float | None:
    '''Get the weighted average bid for the year for a given project item. If unable to find item, return None.'''
    matched_item = None
    matched_item_year = None
    for search_wrapper in item_search_wrappers:
        search_tool = search_wrapper.create_search_tool()
        search_agent = Agent(
            name=f'"{project_item.item_name}" Item Search Agent',
            instructions=search_agent_prompt,
            tools=[search_tool, report_item_choice, report_failure_to_find_item],
            tool_use_behavior={'stop_at_tool_names': [report_item_choice.name, report_failure_to_find_item.name]},
            reset_tool_choice=False,
            model=model
        )
        agent_input = [
            {
                'role': 'user',
                'content': f"Find the weighted average bid (year) for the following project item:\n\nItem Number: {project_item.item_number}\nItem Name: {project_item.item_name}\nUnit: {project_item.unit}\nQuantity: {project_item.quantity}\n\nCall the report_item_choice tool to report your result. If you cannot find the item, call the report_failure_to_find_item tool with an explanation."
            }
        ]
        search_result = await Runner.run(search_agent, input=agent_input, run_config=RunConfig(
            workflow_name="(EE) Item search workflow"))  # todo max turns (incl in prompt)


        for i, item in enumerate(search_result.new_items):
            if (
                    i > 0 and
                    isinstance(item, agents.items.ToolCallOutputItem) and
                    isinstance(search_result.new_items[i-1], agents.items.ToolCallItem)
            ):
                if search_result.new_items[i-1].raw_item.name == report_item_choice.name:
                    item_choice = item.output
                    matched_item = search_wrapper.choice_to_item(item_choice)
                    matched_item_year = search_wrapper.year
                    break
                elif search_result.new_items[i-1].raw_item.name == report_failure_to_find_item.name:
                    break
        if matched_item is not None:
            break
    if matched_item is None:
        return None, None

    wab_extraction_inputs = [
        {
            'role': 'system',
            'content': extract_wab_prompts[matched_item_year]
        },
        {
            'role': 'user',
            'content': '\n'.join(matched_item['lines'])
        }
    ]
    wab_extraction_response = await async_client.responses.parse(
        input = wab_extraction_inputs,
        model = model,
        text_format = WeightedAverageBid
    )

    wab_float = wab_extraction_response.output_parsed.weighted_average_average_bid_for_year
    return wab_float, matched_item_year

async def batch_item_weighted_avg_bids(
        items_list: ProjectItemsList,
        item_search_wrappers: list[ItemSearchWrapper],
        search_agent_prompt: str, extract_wab_prompts: dict[int, str],
        async_client: AsyncOpenAI, model: str = 'gpt-5'
) -> float | None:
    wab_coroutines = []
    for project_item in items_list.items:
        wab_coroutines.append(get_item_weighted_avg_bid(project_item, item_search_wrappers, search_agent_prompt, extract_wab_prompts, async_client, model=model))
    return await asyncio.gather(*wab_coroutines)

if __name__ == '__main__':
    with open('../changeorders/config.toml', 'rb') as f:
        cmo_config = tomli.load(f)

    with open('config.toml', 'rb') as f:
        config = tomli.load(f)
    with open(config['extract_project_items_prompt_path'], 'r', encoding='utf-8') as f:
        extract_project_items_prompt = f.read()
    with open(config['search_agent_prompt_path'], 'r', encoding='utf-8') as f:
        search_agent_prompt = f.read()
    with open(config['extract_wab_prompt_path'], 'r', encoding='utf-8') as f:
        extract_wab_prompt = f.read()

    cost_items_lists = []
    for cost_book_txt_path in cmo_config['cost_book_txt_paths']:
        with open(cost_book_txt_path['path'], 'r') as f:
            cost_book_text = f.read()
        if cost_book_txt_path['year'] == 2024:
            cost_items = get_cost_items(cost_book_text, start_line=cost_book_txt_path['start_line'])
        else:
            cost_items = old_get_cost_items(cost_book_text, start_line=cost_book_txt_path['start_line'])
        cost_items_lists.append([cost_items, cost_book_txt_path['year']])
    cost_items_lists = sorted(cost_items_lists, key=lambda x: x[1], reverse=True) # sort by year, most recent first

    with open('../proposalwriting/ddl_config.toml', 'rb') as f:
        ddl_config = tomli.load(f)

    OPENAI_FILES_CACHE_PATH = ddl_config['openai_files_cache_path']
    OPENAI_API_KEY = ddl_config['openai_api_key']
    set_default_openai_key(OPENAI_API_KEY)
    async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    PROJECT_ITEMS_TABLE_PATH = config['project_items_table_path']

    project_items_file_id = asyncio.run(get_or_upload_async(PROJECT_ITEMS_TABLE_PATH, async_client, OPENAI_FILES_CACHE_PATH,
                                                      purpose="user_data"))

    extraction_inputs = [
        {
            'role': 'system',
            'content': extract_project_items_prompt
        },
        {
            'role': 'user',
            'content': [{
                'type': 'input_file',
                'file_id': project_items_file_id
            }]
        }
    ]

    extraction_response = asyncio.run(async_client.responses.parse(
        model=config['model'],
        input=extraction_inputs,
        text_format=ProjectItemsList
    ))
    project_items_list = extraction_response.output_parsed

    item_search_wrappers = [ItemSearchWrapper(cost_items, year) for cost_items, year in cost_items_lists]
    wabs_n_years = asyncio.run(batch_item_weighted_avg_bids(project_items_list, item_search_wrappers, search_agent_prompt,
                                           extract_wab_prompt, async_client, model=config['model']))

    total_engineers_estimate = 0.
    excluded_items = []
    for item, (wab, year) in zip(project_items_list.items, wabs_n_years):
        if wab is None:
            print(f"Failed to find weighted average bid for item {item.item_number} ({item.item_name})")
            excluded_items.append(item)
        else:
            print(f"Item {item.item_number} ({item.item_name}), from year {year}: Weighted Average Bid for Year = {wab}")
            total_engineers_estimate += wab * item.quantity
    print(f"Total Engineer's Estimate: {total_engineers_estimate:.2f}")
    if len(excluded_items) > 0:
        print("Excluded items:")
        for item in excluded_items:
            print(f"- {item.item_number} ({item.item_name})")