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
import base64
import warnings
from functools import partial
import pickle
import shutil
import os
import time


from estimate_cdot_utils import ProjectItem, ProjectItemsList, ItemSearchWrapper, batch_item_wabs_n_cdot_names
from estimate_global_utils import async_whisper_pdf_text_extraction, get_or_upload_async

def confirm_items_match(item1: ProjectItem, item2: ProjectItem, item_name_match_threshold: float = 80.) -> bool:
    '''Raise an error if item1 (ProjectItem) and item2 (dict) do not match in name and unit.'''
    if item1.item_number.strip().lower() != item2.item_number.strip().lower():
        return False
    if item1.unit.strip().lower() != item2.unit.strip().lower():
        return False
    if fuzz.ratio(item1.item_name.lower().strip(), item2.item_name.lower().strip()) < item_name_match_threshold:
        return False
    return True

def match_project_items(openai_project_items_list: list[ProjectItem], claude_project_items_list: list[ProjectItem], item_name_match_threshold: float = 80.):
    '''Compare two lists of project items extracted by different models, and return a tuple of (matched_items, disputed_items, openai_unmatched_items, claude_unmatched_items)'''
    scores = []
    for openai_idx, openai_item in enumerate(openai_project_items_list):
        for claude_idx, claude_item in enumerate(claude_project_items_list):
            score = fuzz.ratio(openai_item.item_number.lower().strip(), claude_item.item_number.lower().strip())
            score += fuzz.ratio(openai_item.item_name.lower().strip(), claude_item.item_name.lower().strip())
            score += fuzz.ratio(openai_item.unit.lower().strip(), claude_item.unit.lower().strip())
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
            if abs(openai_item.quantity - claude_item.quantity) > 0.1:
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
        claude_item: ProjectItem,
        project_items_pdf_base_64: str,
        openai_project_items_file_id: str,
        openai_async_client: AsyncOpenAI, openai_model: str,
        async_claude_client: AsyncAnthropic, claude_model: str,
        item_name_match_threshold: float = 80.,
        project_items_table_ocr_str: str|None=None
):
    item_name = None
    if fuzz.ratio(openai_item.item_name.lower().strip(),
                  claude_item.item_name.lower().strip()) >= item_name_match_threshold:
        item_name = openai_item.item_name
    item_number = None
    if openai_item.item_number == claude_item.item_number:
        item_number = openai_item.item_number
    unit = None
    if openai_item.unit == claude_item.unit:
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
                        'data': project_items_pdf_base_64
                    }
                }
            ]
        },
        {
            'role': 'assistant',
            'content': '{'
        }
    ]
    claude_check_couroutine = async_claude_client.messages.create(
        model=claude_model,
        messages=claude_check_inputs,
        max_tokens=10_000
    )

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
                    'file_id': openai_project_items_file_id
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
            openai_check_item.item_number == claude_check_item.item_number
            and fuzz.ratio(openai_check_item.item_name.strip().lower(), claude_check_item.item_name.strip().lower()) >= item_name_match_threshold
            and openai_check_item.unit.strip().lower() == claude_check_item.unit.strip().lower()
            and abs(openai_check_item.quantity - claude_check_item.quantity) < 0.1
    ):
        return True, openai_check_item
    else:
        final_item = {}
        if openai_check_item.item_number == claude_check_item.item_number:
            final_item['item_number'] = openai_check_item.item_number
        if fuzz.ratio(openai_check_item.item_name.strip().lower(), claude_check_item.item_name.strip().lower()) >= item_name_match_threshold:
            final_item['item_name'] = openai_check_item.item_name
        if openai_check_item.unit.strip().lower() == claude_check_item.unit.strip().lower():
            final_item['unit'] = openai_check_item.unit
        if abs(openai_check_item.quantity - claude_check_item.quantity) < 0.1:
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
    claude_project_items_json = json.loads('['+claude_extraction_response.content[0].text)
    claude_project_items_list = [ProjectItem.model_validate(item) for item in claude_project_items_json]

    matched_items, disputed_items, openai_unmatched_items, claude_unmatched_items = match_project_items(openai_project_items_list, claude_project_items_list, item_name_match_threshold=item_name_match_threshold)

    final_undecided_items = [item.model_dump() for item in claude_unmatched_items + openai_unmatched_items]

    check_disputed_pair_partial = partial(
        check_disputed_pair,
        project_items_pdf_base_64 = base64_pdf_string,
        openai_project_items_file_id = project_items_file_id,
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



class Estimator:
    def __init__(
            self,
            project_items_table_path: str,
            unstract_api_key: str, openai_api_key: str,
            async_openai_client: AsyncOpenAI, openai_files_cache_path: str, openai_extract_project_items_prompt: str, openai_model: str,
            async_claude_client: AsyncAnthropic, claude_extract_project_items_prompt: str, claude_model: str,
            cost_items_lists: dict,
            search_agent_prompt: str,
            extract_wab_prompts: dict[int, str],
            wab_cache_path: str,
            item_name_match_threshold: float = 80.,
    ):
        self.project_items_table_path = project_items_table_path
        self.unstract_api_key = unstract_api_key
        self.openai_api_key = openai_api_key
        self.async_openai_client = async_openai_client
        self.openai_files_cache_path = openai_files_cache_path
        self.openai_extract_project_items_prompt = openai_extract_project_items_prompt
        self.openai_model = openai_model
        self.async_claude_client = async_claude_client
        self.claude_extract_project_items_prompt = claude_extract_project_items_prompt
        self.claude_model = claude_model
        self.cost_items_lists = cost_items_lists
        self.search_agent_prompt = search_agent_prompt
        self.extract_wab_prompts = extract_wab_prompts
        self.wab_cache_path = wab_cache_path
        self.item_name_match_threshold = item_name_match_threshold

        self.table_unstract_json_response = None
        self.table_unstract_ocr_text = None

    async def get_wabs(self):
        agents.set_default_openai_key(self.openai_api_key)

        project_items, undecided_items = await self.get_project_items()
        project_items_list = ProjectItemsList(items=project_items)

        item_search_wrappers = [ItemSearchWrapper(cost_items, year) for year, cost_items in self.cost_items_lists.items()]
        item_info_dicts = await batch_item_wabs_n_cdot_names(
            project_items_list,
            item_search_wrappers,
            self.search_agent_prompt,
            self.extract_wab_prompts,
            self.async_openai_client,
            model=self.openai_model,
            use_cache = True,
            cache_path = self.wab_cache_path
        )

        out_items = [
            {
                'cdot_name': item_dict['cdot_name'],
                'wab_float': item_dict['wab_float'],
                'matched_year': item_dict['matched_year'],
                'ProjectItem': item
            }
            for item_dict, item in zip(item_info_dicts, project_items_list.items)
        ]

        return out_items, undecided_items


    async def get_table_unstract_json(self):
        self.table_unstract_json_response = await async_whisper_pdf_text_extraction(unstract_api_key=self.unstract_api_key,input_pdf_path=self.project_items_table_path, return_json=True, add_line_nos=False)
        self.table_unstract_ocr_text = self.table_unstract_json_response['result_text']
        return self.table_unstract_json_response

    async def get_project_items(self):
        if self.table_unstract_json_response is None:
            await self.get_table_unstract_json()

        # openai extraction
        self.openai_table_file_id = await get_or_upload_async(self.project_items_table_path, self.async_openai_client,
                                                          self.openai_files_cache_path, purpose="user_data")
        openai_extraction_inputs = [
            {
                'role': 'system',
                'content': self.openai_extract_project_items_prompt
            },
            {
                'role': 'user',
                'content': [{
                    'type': 'input_file',
                    'file_id': self.openai_table_file_id
                }]
            }
        ]
        if self.table_unstract_ocr_text is not None:
            openai_extraction_inputs[1]['content'].append({
                'type': 'input_text',
                'text': f'The following is the OCR text extracted from the PDF, which you should cross-reference with the table: \n{self.table_unstract_ocr_text}'
            })
        openai_extraction_coroutine = self.async_openai_client.responses.parse(
            model=self.openai_model,
            input=openai_extraction_inputs,
            text_format=ProjectItemsList,
            metadata={'source': 'ee_project_items_extraction'}
        )

        # claude extraction
        with open(self.project_items_table_path, 'rb') as f:
            pdf_data = f.read()
        base64_pdf_data = base64.standard_b64encode(pdf_data)
        self.project_items_pdf_base64 = base64_pdf_data.decode('utf-8')
        claude_extraction_inputs = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': self.claude_extract_project_items_prompt
                    },
                    {
                        'type': 'document',
                        'source': {
                            'type': 'base64',
                            'media_type': 'application/pdf',
                            'data': self.project_items_pdf_base64
                        }
                    }
                ]
            },
            {
                'role': 'assistant',
                'content': '['
            }
        ]
        if self.table_unstract_ocr_text is not None:
            claude_extraction_inputs[0]['content'].append({
                'type': 'text',
                'text': f'The following is the OCR text extracted from the PDF, which you should cross-reference with the table: \n{self.table_unstract_ocr_text}'
            })
        claude_extraction_couroutine = self.async_claude_client.messages.create(model=self.claude_model,
                                                                           messages=claude_extraction_inputs,
                                                                           max_tokens=10_000)

        openai_extraction_response, claude_extraction_response = await asyncio.gather(openai_extraction_coroutine,
                                                                                      claude_extraction_couroutine)
        openai_project_items_list = openai_extraction_response.output_parsed.items
        claude_project_items_json = json.loads('[' + claude_extraction_response.content[0].text)
        claude_project_items_list = [ProjectItem.model_validate(item) for item in claude_project_items_json]

        matched_items, disputed_items, openai_unmatched_items, claude_unmatched_items = match_project_items(
            openai_project_items_list, claude_project_items_list, item_name_match_threshold=self.item_name_match_threshold)

        final_undecided_items = [item.model_dump() for item in claude_unmatched_items + openai_unmatched_items]

        disputed_results = await asyncio.gather(*[
            self.check_disputed_pair(openai_item=openai_item, claude_item=claude_item)
            for openai_item, claude_item in disputed_items
        ])

        for is_match, final_item in disputed_results:
            if is_match:
                matched_items.append(final_item)
            else:
                final_undecided_items.append(final_item)
        return matched_items, final_undecided_items

    async def check_disputed_pair(
            self,
            openai_item: ProjectItem,
            claude_item: ProjectItem,
    ):
        item_name = None
        if fuzz.ratio(openai_item.item_name.lower().strip(),
                      claude_item.item_name.lower().strip()) >= self.item_name_match_threshold:
            item_name = openai_item.item_name
        item_number = None
        if openai_item.item_number == claude_item.item_number:
            item_number = openai_item.item_number
        unit = None
        if openai_item.unit == claude_item.unit:
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
                            'data': self.project_items_pdf_base64
                        }
                    }
                ]
            },
            {
                'role': 'assistant',
                'content': '{'
            }
        ]
        claude_check_couroutine = self.async_claude_client.messages.create(
            model=self.claude_model,
            messages=claude_check_inputs,
            max_tokens=10_000
        )

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
                        'file_id': self.openai_table_file_id
                    }]
            }
        ]
        openai_check_coroutine = self.async_openai_client.responses.parse(
            model=self.openai_model,
            input=openai_check_inputs,
            text_format=ProjectItem,
            metadata={'source': 'ee_project_items_checking'}
        )
        check_results = await asyncio.gather(openai_check_coroutine, claude_check_couroutine)
        openai_check_item = check_results[0].output_parsed
        claude_check_item = json.loads('{'+check_results[1].content[0].text)
        if (
                openai_check_item.item_number == claude_check_item.item_number
                and fuzz.ratio(openai_check_item.item_name.strip().lower(), claude_check_item.item_name.strip().lower()) >= self.item_name_match_threshold
                and openai_check_item.unit.strip().lower() == claude_check_item.unit.strip().lower()
                and abs(openai_check_item.quantity - claude_check_item.quantity) < 0.1
        ):
            return True, openai_check_item
        else:
            final_item = {}
            if openai_check_item.item_number == claude_check_item.item_number:
                final_item['item_number'] = openai_check_item.item_number
            if fuzz.ratio(openai_check_item.item_name.strip().lower(), claude_check_item.item_name.strip().lower()) >= item_name_match_threshold:
                final_item['item_name'] = openai_check_item.item_name
            if openai_check_item.unit.strip().lower() == claude_check_item.unit.strip().lower():
                final_item['unit'] = openai_check_item.unit
            if abs(openai_check_item.quantity - claude_check_item.quantity) < 0.1:
                final_item['quantity'] = openai_check_item.quantity
            return False, final_item

if __name__ == '__main__':
    with open('../GlobalUtils/config.toml', 'rb') as f:
        global_config = tomli.load(f)
    with open('../CDOTCostData/config.toml', 'rb') as f:
        cost_data_config = tomli.load(f)

    with open('config.toml', 'rb') as f:
        config = tomli.load(f)
    with open(config['openai_extract_project_items_prompt_path'], 'r', encoding='utf-8') as f:
        openai_extract_project_items_prompt = f.read()
    with open(config['claude_extract_project_items_prompt_path'], 'r', encoding='utf-8') as f:
        claude_extract_project_items_prompt = f.read()
    with open(config['search_agent_prompt_path'], 'r', encoding='utf-8') as f:
        search_agent_prompt = f.read()
    extract_wab_prompts = {}
    for wab_prompt_ref in config['extract_wab_prompts_paths']:
        with open(wab_prompt_ref['path'], 'r', encoding='utf-8') as f:
            extract_wab_prompts[wab_prompt_ref['year']] = f.read()

    cost_items_lists = []
    for cost_book_ref in cost_data_config['cost_data_books']:
        with open(cost_book_ref['cost_items_pkl_path'], 'rb') as f:
            cost_items = pickle.load(f)
        cost_items_lists.append([cost_items, cost_book_ref['year']])

    # cost_items_lists = sorted(cost_items_lists, key=lambda x: x[1],
    #                                               reverse=True)  # sort by year, most recent first
    cost_items_lists = {year: items for items, year in cost_items_lists}  # convert to dict for easy access

    OPENAI_FILES_CACHE_PATH = global_config['openai_files_cache_path']
    OPENAI_API_KEY = global_config['openai_api_key']
    async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    PROJECT_ITEMS_TABLE_PATH = config['project_items_table_path']

    my_estimator = Estimator(
        project_items_table_path=PROJECT_ITEMS_TABLE_PATH,
        unstract_api_key=global_config['unstract_api_key'],
        openai_api_key=OPENAI_API_KEY,
        async_openai_client=async_client,
        openai_files_cache_path=OPENAI_FILES_CACHE_PATH,
        openai_extract_project_items_prompt=openai_extract_project_items_prompt,
        openai_model=config['openai_model'],
        async_claude_client=AsyncAnthropic(api_key=global_config['anthropic_api_key']),
        claude_extract_project_items_prompt =claude_extract_project_items_prompt,
        claude_model=config['claude_model'],
        cost_items_lists=cost_items_lists,
        search_agent_prompt=search_agent_prompt,
        extract_wab_prompts=extract_wab_prompts,
        wab_cache_path=cost_data_config['wab_cache_json_path'],
    )

    out_items, undecided_items = asyncio.run(my_estimator.get_wabs())
    total_engineers_estimate = 0.
    excluded_items = []
    for item in out_items:
        wab = item['wab_float']
        year = item['matched_year']
        item_obj = item['ProjectItem']
        if wab is None:
            print(f"Failed to find weighted average bid for item {item_obj.item_number} ({item_obj.item_name})")
            excluded_items.append(item_obj)
        else:
            print(f"Item {item_obj.item_number} ({item_obj.item_name}), from year {item['matched_year']}: Weighted Average Bid for Year = {item['wab_float']}")
            total_engineers_estimate += wab * item_obj.quantity
    print(f"Total Engineer's Estimate: {total_engineers_estimate:.2f}")
    if len(excluded_items) > 0:
        print("Excluded items:")
        for item in excluded_items:
            print(f"- {item.item_number} ({item.item_name})")


    # project_items_file_id = asyncio.run(get_or_upload_async(PROJECT_ITEMS_TABLE_PATH, async_client, OPENAI_FILES_CACHE_PATH,
    #                                                   purpose="user_data"))
    #
    # extraction_inputs = [
    #     {
    #         'role': 'system',
    #         'content': extract_project_items_prompt
    #     },
    #     {
    #         'role': 'user',
    #         'content': [{
    #             'type': 'input_file',
    #             'file_id': project_items_file_id
    #         }]
    #     }
    # ]
    #
    # extraction_response = asyncio.run(async_client.responses.parse(
    #     model=config['model'],
    #     input=extraction_inputs,
    #     text_format=ProjectItemsList
    # ))
    # project_items_list = extraction_response.output_parsed
    #
    # item_search_wrappers = [ItemSearchWrapper(cost_items, year) for cost_items, year in cost_items_lists]
    # wabs_n_years = asyncio.run(batch_item_weighted_avg_bids(project_items_list, item_search_wrappers, search_agent_prompt,
    #                                        extract_wab_prompt, async_client, model=config['model']))