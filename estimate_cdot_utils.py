import asyncio
import os
from tempfile import TemporaryDirectory
from typing import Callable, Optional
import fitz
from openai import AsyncOpenAI
from pydantic import BaseModel
from rapidfuzz import fuzz, process
from fuzzysearch import find_near_matches
from agents import function_tool
import json
import hashlib
import warnings
from datetime import datetime
from PIL import Image
from pathlib import Path
import agents
from agents import Agent, Runner, RunConfig

from estimate_global_utils import async_whisper_pdf_text_extraction, get_unstract_citation_images

# cdotcostdata.cost_data_utils
class ProjectItem(BaseModel):
    item_number: str
    item_name: str
    unit: str
    quantity: int | float

class ProjectItemsList(BaseModel):
    items: list[ProjectItem]

class WeightedAverageBid(BaseModel):
    weighted_average_average_bid_for_year: float | None

class ItemChoice(BaseModel):
    item_id: int
    item_number: str
    item_name: str


# cdotcostdata.cost_data_utils
class ItemSearchWrapper:
    def __init__(
            self,
            cost_items: list[dict],
            year: int,
            n_matches: int = 5,
            n_sub_matches: int = 20
    ):
        self.cost_items = cost_items
        self.year = year
        self.n_matches = n_matches
        self.n_sub_matches = n_sub_matches
        self.item_number_ids = dict()
        self.id_to_item = dict()

    def search_cost_items(self, number_query = None, name_query = None):
        '''Search for items in cost_items by code or item name using fuzzy matching.

        Returns a list of cost items sorted by score in descending order.'''
        assert number_query is not None or name_query is not None, "At least one of number_query or name_query must be provided"
        item_scores = dict() # maps item index to score
        if number_query is not None:
            code_entries = [item['item_number'] for item in self.cost_items]
            best_matches = process.extract(number_query, code_entries, scorer=fuzz.WRatio, limit=self.n_sub_matches)
            for match in best_matches:
                item_scores[match[2]] = match[1]

        if name_query is not None:
            name_entries = [item['name'] for item in self.cost_items]
            best_matches = process.extract(name_query, name_entries, scorer=fuzz.WRatio, limit=self.n_sub_matches)
            for match in best_matches:
                if match[2] in item_scores:
                    item_scores[match[2]] += match[1]
                else:
                    item_scores[match[2]] = match[1]

        best_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:self.n_matches]
        return [self.cost_items[item_idx] for item_idx, score in best_scores]

    def create_search_tool(self, tool_name='search_cost_items_tool'):
        @function_tool(
            name_override=tool_name,
            failure_error_function=None
        )
        def search_cost_items_tool(number_query: str | None = None, name_query: str | None = None) -> list[dict]:
            '''Search for items in cost_items by code or item name using fuzzy matching.

            Returns a list of results, each with an index, item name, and item number.'''
            item_results = self.search_cost_items(number_query, name_query)
            out_results = []
            for item in item_results:
                if item['item_number'] in self.item_number_ids:
                    item_id = self.item_number_ids[item['item_number']]
                else:
                    item_id = len(self.item_number_ids)
                    self.item_number_ids[item['item_number']] = item_id
                    self.id_to_item[item_id] = item
                out_results.append({'item_id': item_id, 'item_name': item['name'], 'item_number': item['item_number'], 'year': item['year'], 'unit': item['unit']})
            return out_results
        return search_cost_items_tool

    def choice_to_item(self, selected_item, threshold=90):
        """
        Take an ItemChoice and return a dict representing the cost item

        Args:
            selected_item: ItemChoice with 'item_id', 'item_number', 'item_name' attributes
            threshold: Minimum similarity score (0-100) to consider a match

        Returns:
            dict: The matched cost item if found, else None
        """
        item_id = selected_item.item_id
        if item_id not in self.id_to_item:
            raise ValueError(f"ID {item_id} not found in cost items")
        actual_result = self.id_to_item[item_id]

        # Fuzzy match item number
        number_similarity = fuzz.ratio(
            selected_item.item_number,
            actual_result['item_number']
        )

        # Fuzzy match item name
        name_similarity = fuzz.ratio(
            selected_item.item_name,
            actual_result['name']
        )

        # Pass if either field meets threshold
        if number_similarity < threshold and name_similarity < threshold:
            raise ValueError("Item at with selected ID does not closely match any known cost item")
        else:
            return actual_result


@function_tool
def report_item_choice(item_choice: ItemChoice) -> str:
    '''Report the item result corresponding to the weighted average bid for the year.'''
    return item_choice

@function_tool
def report_failure_to_find_item(explanation: str) -> str:
    '''Report failure to find item in cost data.'''
    return explanation

def get_wab_cache_key(project_item: ProjectItem) -> str:
    '''Generate a cache key for a project item.'''
    return f"{project_item.item_number}||{project_item.item_name}||{project_item.unit}".lower().strip()

def save_wab_cache(cache_path: str, cache_data: dict):
    '''Save the weighted average bid cache to a JSON file.'''
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)

def load_wab_cache(cache_path: str) -> dict:
    '''Load the weighted average bid cache from a JSON file.'''
    if not Path(cache_path).exists():
        print(f"Could not find WAB cache at: {cache_path}")
        return dict()
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    return cache_data

def update_wab_cache(
        project_item: ProjectItem,
        item_info_dict: dict,
        cache_path: str
):
    '''Update the weighted average bid cache with a new item info dict.'''
    cache = load_wab_cache(cache_path)
    cache_key = get_wab_cache_key(project_item)
    cache[cache_key] = item_info_dict
    if cache_path is not None:
        save_wab_cache(cache_path, cache)

def check_wab_cache(
        project_item: ProjectItem,
        cache_path: str
) -> dict | None:
    '''Check the weighted average bid cache for a project item. Return the item info dict if found, else None.'''
    cache = load_wab_cache(cache_path)
    cache_key = get_wab_cache_key(project_item)
    if cache_key in cache:
        return cache[cache_key]
    else:
        return None

#cdotcostdata.cost_data_citation
def generate_cache_key(cost_item: dict) -> str:
    """
    Generate a unique cache key for a cost item.

    Uses a hash of the item's metadata to ensure uniqueness.

    Args:
        cost_item: Dictionary containing cost item metadata

    Returns:
        A hexadecimal hash string to use as cache key
    """
    # Create a deterministic string from cost item metadata
    key_data = {
        'item_number': cost_item.get('item_number', ''),
        'year': cost_item.get('year', ''),
        'name': cost_item.get('name', ''),
        'unit': cost_item.get('unit', '')
    }

    # Sort keys to ensure consistent ordering
    key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=False)

    # Generate SHA256 hash
    return hashlib.sha256(key_string.encode()).hexdigest()


def load_cache(cache_json_path: str) -> dict:
    """
    Load the cache mapping from disk.

    Args:
        cache_json_path: Path to the cache JSON file

    Returns:
        Dictionary mapping cache keys to cache entries
    """
    if not os.path.exists(cache_json_path):
        warnings.warn('Cache JSON file not found.')
        return {}

    try:
        with open(cache_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load cache from {cache_json_path}: {e}")
        return {}


def save_cache(cache_data: dict, cache_json_path: str) -> None:
    """
    Save the cache mapping to disk.

    Args:
        cache_data: Dictionary mapping cache keys to cache entries
        cache_json_path: Path to the cache JSON file
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_json_path), exist_ok=True)

    try:
        with open(cache_json_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Could not save cache to {cache_json_path}: {e}")


def get_cached_images(
        cost_item: dict,
        cache_json_path: str
):
    """
    Retrieve cached citation images for a cost item if they exist.

    Args:
        cost_item: Dictionary containing cost item metadata
        cache_json_path: Path to the cache JSON file

    Returns:
        None if cache miss, List of PIL Images or tuple of (images (list), page_numbers (list)) if cache hit
    """
    cache_key = generate_cache_key(cost_item)
    cache_data = load_cache(cache_json_path)
    if cache_key not in cache_data:
        return None
    cache_entry = cache_data[cache_key]
    image_paths = cache_entry.get('images', [])
    page_numbers = cache_entry.get('page_numbers', [])
    if len(page_numbers) != len(image_paths):
        warnings.warn("Mismatch between number of cached images and page numbers")
        return None

    # Verify all image files still exist
    if not all(os.path.exists(path) for path in image_paths):
        print(f"Warning: Some cached images missing for cache key {cache_key}")
        return None

    # Load images from disk
    try:
        images = [Image.open(path) for path in image_paths]
        return images, page_numbers
    except Exception as e:
        print(f"Warning: Could not load cached images: {e}")
        return None


def save_images_to_cache(
        images: list,
        page_numbers: list[int],
        cost_item: dict,
        cache_json_path: str,
        cache_dir: str,
) -> None:
    """
    Save citation images to cache.

    Args:
        images: List of PIL Images to cache, or list of tuples of (PIL Image, page_number)
        page_numbers: List of page numbers corresponding to each image
        cost_item: Dictionary containing cost item metadata
        cache_json_path: Path to the cache JSON file
        cache_dir: Directory to save image files
    """
    cache_key = generate_cache_key(cost_item)

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Save each image as PNG
    image_paths = []
    for idx, img in enumerate(images):
        filename = f"{cache_key}_{idx}.png"
        filepath = os.path.join(cache_dir, filename)

        try:
            img.save(filepath, format='PNG')
            image_paths.append(filepath)
        except Exception as e:
            print(f"Warning: Could not save image {idx} for cache key {cache_key}: {e}")
            # Clean up partially saved images
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            return

    # Update cache mapping
    cache_data = load_cache(cache_json_path)
    cache_data[cache_key] = {
        'item_number': cost_item.get('item_number', ''),
        'year': cost_item.get('year', ''),
        'name': cost_item.get('name', ''),
        'images': image_paths,
        'page_numbers': page_numbers,
        'cached_at': datetime.now().isoformat()
    }

    save_cache(cache_data, cache_json_path)



#cdotcostdata.cost_data_utils
async def get_item_wab_n_cdot_name(
        project_item: ProjectItem,
        item_search_wrappers: list[ItemSearchWrapper], # in descending chronological order, most recent first
        search_agent_prompt: str, extract_wab_prompts: dict[int, str],
        async_client: AsyncOpenAI, model: str = 'gpt-5',
        use_cache: bool = False,
        cache_path: str | None = None
) -> dict:
    '''Get the weighted average bid for the year for a given project item. If unable to find item, return None.'''
    if use_cache:
        try:
            cached_result = check_wab_cache(project_item, cache_path)
            if cached_result is not None:
                print(f'WAB cache hit for item: {project_item.item_name}')
                return cached_result
            print(f'WAB cache miss for item: {project_item.item_name}')
        except Exception as e:
            print(f"Error checking WAB cache: {e}")
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
        return {
            'cdot_name': None,
            'wab_float': None,
            'matched_year': None
        }

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
    item_info_dict = {
        'cdot_name': matched_item['name'],
        'wab_float': wab_float,
        'matched_year': matched_item_year
    }
    if use_cache:
        try:
            update_wab_cache(project_item, item_info_dict, cache_path)
        except Exception as e:
            print(f"Error updating WAB cache: {e}")
    return item_info_dict

# cdotcostdata.cost_data_utils
async def batch_item_wabs_n_cdot_names(
        items_list: ProjectItemsList,
        item_search_wrappers: list[ItemSearchWrapper],
        search_agent_prompt: str, extract_wab_prompts: dict[int, str],
        async_client: AsyncOpenAI, model: str = 'gpt-5',
        use_cache: bool = False,
        cache_path: str | None = None
) -> float | None:
    wab_coroutines = []
    for project_item in items_list.items:
        wab_coroutines.append(
            get_item_wab_n_cdot_name(
                project_item, item_search_wrappers,
                search_agent_prompt, extract_wab_prompts,
                async_client, model=model,
                use_cache = use_cache, cache_path = cache_path
            )
        )
    return await asyncio.gather(*wab_coroutines)




# cdotcostdata.cost_data_citation
def find_matching_pages(
        doc: fitz.Document,
        query: str,
        max_l_dist: int | None = None,
        text_preprocessor: Callable[[str], str] | None = None,
) -> list[int]:
    if max_l_dist is None:
        max_l_dist = max(1, len(query) // 5)

    matching_pages = set()
    if text_preprocessor is not None:
        query = text_preprocessor(query)
    wab_pattern = 'Weighted Average for the Year'
    if text_preprocessor is not None:
        wab_pattern = text_preprocessor(wab_pattern)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        if text_preprocessor is not None:
            page_text = text_preprocessor(page_text)
        matches = find_near_matches(
            query,
            page_text,
            max_l_dist = max_l_dist
        )
        if matches:
            first_match_end = sorted(matches, key = lambda m: m.start)[0].end
            if find_near_matches(wab_pattern, page_text[first_match_end:], max_l_dist = 1):
                matching_pages.add(page_num)
            else: # check next few pages for weighted average bid for the year pattern
                max_page = min(len(doc), page_num + 5)
                for page_end in range(page_num + 1, max_page):
                    next_page = doc.load_page(page_end)
                    next_page_text = next_page.get_text()
                    if text_preprocessor is not None:
                        next_page_text = text_preprocessor(next_page_text)
                    if page_end == len(doc)-1 or find_near_matches(wab_pattern, next_page_text, max_l_dist = 1):
                        matching_pages.update(range(page_num, page_end+1))
                        break
    return list(matching_pages)

# cdotcostdata.cost_data_citation
def find_item_pages(
        doc: fitz.Document,
        item: dict,
        text_preprocessor: Callable[[str], str] | None = None,
) -> list[int]:

    #TODO - Remove and reinstate below? This uses UNION of name and item number pages, whereas below uses intersection
    name_pages = set(find_matching_pages(doc, item['name'], text_preprocessor=text_preprocessor))
    item_no_pages = find_matching_pages(doc, item['item_number'], text_preprocessor=text_preprocessor, max_l_dist=1)
    name_pages.update(item_no_pages)
    return list(name_pages)

    # name_pages = set(find_matching_pages(doc, item['name'], text_preprocessor = text_preprocessor))
    # item_no_pages = find_matching_pages(doc, item['item_number'], text_preprocessor = text_preprocessor)
    # item_pages = []
    # for page in item_no_pages:
    #     if page in name_pages:
    #         item_pages.appen

# cdotcostdata.cost_data_citation
async def extract_item_citation_images_pages(
        cost_item: dict,
        input_pdf_path: str,
        citation_prompt: str,
        unstract_api_key: str,
        openai_client: AsyncOpenAI,
        text_preprocessor: Callable[[str], str] | None = None,
        use_cache: bool = False,
        cache_json_path: Optional[str] = None,
        cache_dir: Optional[str] = None
):
    # Check cache if enabled
    if use_cache:
        assert cache_json_path is not None, "Cache JSON path must be provided if use_cache is True"
        assert cache_dir is not None, "Cache directory must be provided if use_cache is True"
        cache_result = get_cached_images(cost_item, cache_json_path)
        if cache_result is not None:
            print(f"(CDOT Citation) Cache hit for item {cost_item.get('item_number', 'unknown')}")
            return cache_result
        print(f"(CDOT Citation) Cache miss for item {cost_item.get('item_number', 'unknown')}")

    source_doc = fitz.open(input_pdf_path)
    item_pages = find_item_pages(source_doc, cost_item, text_preprocessor = text_preprocessor)
    if not item_pages:
        return None

    item_doc = fitz.Document() # document to hold item pages
    for page_num in item_pages:
        item_doc.insert_pdf(source_doc, from_page=page_num, to_page=page_num)
    source_doc.close()
    print(f'Found subdoc with {len(item_doc)} pages for item "{cost_item['name']}", pdf path:{input_pdf_path}') # TODO REMOVE

    with TemporaryDirectory() as temp_dir:
        item_pdf_path = os.path.join(temp_dir, 'item.pdf')
        item_doc.save(item_pdf_path)
        unstract_json = await async_whisper_pdf_text_extraction(
            unstract_api_key = unstract_api_key,
            input_pdf_path = item_pdf_path,
            return_json=True
        )

    citation_query = f'Item name: {cost_item['name']}, Item number: {cost_item['item_number']}, Item unit: {cost_item['unit']}'
    if 'weighted_average_bid_year' in cost_item:
        citation_query += f', Weighted average bid (year): {cost_item["weighted_average_bid_year"]}'
    citation_images, citation_page_numbers = await get_unstract_citation_images(
        pdf_source = item_doc.tobytes(),
        unstract_response_json = unstract_json,
        citation_query = citation_query,
        citation_prompt = citation_prompt,
        openai_client = openai_client,
        return_page_numbers=True
    )
    citation_page_numbers = [item_pages[p] for p in citation_page_numbers]  # Map back to original PDF page numbers

    # Save to cache if enabled
    if use_cache:
        save_images_to_cache(
            images = citation_images,
            page_numbers = citation_page_numbers,
            cost_item = cost_item,
            cache_json_path = cache_json_path,
            cache_dir=cache_dir
        )
        print(f"Saved images to cache for item {cost_item.get('item_number', 'unknown')}")

    return citation_images, citation_page_numbers