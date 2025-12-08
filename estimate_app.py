from tempfile import TemporaryDirectory
import pandas as pd
import streamlit as st
from streamlit_image_zoom import image_zoom
import tomli
import asyncio
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, OpenAI
from agents import set_default_openai_key, trace
import os
import pickle
import sys
from plyer import notification
import datetime
from functools import partial
import time
import mimetypes
import base64

from estimate_utils import get_project_items
from GlobalUtils.ocr import google_ocr_pdf_text_overlay, whisper_pdf_text_extraction
from GlobalUtils.citation import get_unstract_citation_images
from GlobalUtils.st_file_serving import StreamlitPDFServer
from CDOTCostData.cost_data_utils import (
    ItemSearchWrapper,
    batch_item_wabs_n_cdot_names,
    ProjectItem,
    ProjectItemsList
)
from CDOTCostData.cost_data_citation import extract_item_citation_images_pages

# <editor-fold> Authentication
PASSWORD = st.secrets.get("APP_PASSWORD", "change_me")

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
    notification.notify(
        app_name='Engineer\'s Estimate  App',
        title='Login Event',
        message=f'Login at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.',
        timeout=10
    )


if not st.session_state.auth_ok:
    with st.form("login", clear_on_submit=False):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            if pw == PASSWORD:
                st.session_state.auth_ok = True
            else:
                st.error("Nope.")
    st.stop()
# </editor-fold>

def get_wabs(
        claude_api_key: str,
        openai_api_key: str,
        project_items_table_path: str,
        openai_files_cache_path: str,
        openai_extract_project_items_prompt: str,
        claude_extract_project_items_prompt: str,
        search_agent_prompt: str,
        extract_wab_prompts: dict[int, str],
        cost_items_lists: dict,
        claude_model: str,
        openai_model: str,
        project_items_table_ocr_str: str|None = None
):
    '''Get weighted average bids for project items from a project items table.

    Returns a list of tuples dictionaries with 'ProjectItem', 'wab_float', 'cdot_name', and 'matched_year' keys, and a list of undecided items.
    '''
    set_default_openai_key(openai_api_key)
    openai_async_client = AsyncOpenAI(api_key=openai_api_key)

    async_claude_client = AsyncAnthropic(api_key=claude_api_key)

    project_items_list, final_undecided_items = asyncio.run(get_project_items(
        project_items_table_path,
        openai_async_client, openai_files_cache_path, openai_extract_project_items_prompt, openai_model,
        async_claude_client, claude_extract_project_items_prompt, claude_model,
        project_items_table_ocr_str = project_items_table_ocr_str
    ))
    project_items_list = ProjectItemsList(items = project_items_list)

    item_search_wrappers = [ItemSearchWrapper(cost_items, year) for year, cost_items in cost_items_lists.items()]

    item_info_dicts = asyncio.run(
        batch_item_wabs_n_cdot_names(project_items_list, item_search_wrappers, search_agent_prompt,
                                     extract_wab_prompts, openai_async_client, model=openai_model,
                                     use_cache=True, cache_path=st.session_state['cdot_cost_data_config']['wab_cache_json_path'])
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

    return out_items, final_undecided_items

@st.dialog('Confirm Delete', on_dismiss = 'rerun')
def confirm_delete_dialog(item_name: str, item_index: int):
    st.session_state['editor_key'] += 1  # force refresh of data editor
    st.markdown(f'Are you sure you want to delete item **{item_name}** from the estimate? This action cannot be undone.')
    if st.button('Confirm Delete'):
        st.session_state['wab_items'].pop(item_index)
        st.rerun()
    if st.button('Cancel'):
        st.rerun()

@st.dialog('Add Item Manually')
def add_item_dialog():
    item_name = st.text_input(f'Item name')
    item_number = st.text_input(f'Item number')
    item_unit = st.text_input(f'Unit')
    item_quantity = st.number_input(f'Quantity')
    user_wab = st.number_input(f'Weighted Average Bid (per unit)', width=200)

    addable = (item_name != '') and (item_number != '') and (item_unit != '') and (user_wab > 0)
    if st.button(f'Add "{item_name}" to Estimate', key=f'add_undecided_{i}', disabled=(not addable)):
        new_item = ProjectItem(
            item_name=item_name,
            item_number=item_number,
            unit=item_unit,
            quantity=item_quantity
        )
        new_item_dict = {
            'cdot_name': None,
            'wab_float': user_wab,
            'ProjectItem': new_item
        }
        st.session_state['wab_items'].append(new_item_dict)
        st.rerun()

async def get_estimate_citation_images(
        item: ProjectItem,
        source_year: int,
        wab: float|None = None,
        cdot_name: str|None = None
):
    """Get citation images for an item
    Loads from cache if available, or saves to cache after generation."""
    # Check cache
    cache_key = f'{item.item_name}|{item.item_number}|{item.unit}|{item.quantity}'
    if 'citation_cache' not in st.session_state:
        st.session_state['citation_cache'] = {}
    elif cache_key in st.session_state['citation_cache']:
        saq_citation_images, cost_data_citation_result = st.session_state['citation_cache'][cache_key]
        st.info('Loaded source from cache')
        return saq_citation_images, cost_data_citation_result

    print('Creating citation for item with cdot name: ', cdot_name)

    openai_client = AsyncOpenAI(api_key=st.session_state['ddl_config']['openai_api_key'])
    saq_citation_query = f'Item Name: {item.item_name}, Item Number: {item.item_number}, Unit: {item.unit}, Quantity: {item.quantity}'
    saq_citation_task = get_unstract_citation_images(
        pdf_source=st.session_state['saq_file_bytes'],
        unstract_response_json=st.session_state['unstract_response_json'],
        citation_query=saq_citation_query,
        citation_prompt=st.session_state['citation_prompt'],
        openai_client=openai_client
    )

    if cdot_name:
        name = cdot_name
    else:
        name = item.item_name
    cost_item_dict = {
        'name': name,
        'item_number': item.item_number,
        'unit': item.unit,
        'quantity': item.quantity
    }
    if wab is not None:
        cost_item_dict['weighted_average_bid_year'] = wab
    cost_item_pdf_path = st.session_state['cost_item_pdf_paths'][source_year]
    cost_data_citation_task = extract_item_citation_images_pages(
        cost_item=cost_item_dict,
        input_pdf_path=cost_item_pdf_path,
        citation_prompt=st.session_state['cost_data_citation_prompt'],
        unstract_api_key=st.session_state['config']['unstract_api_key'],
        openai_client=openai_client,
        text_preprocessor=lambda x: x.lower(),
        use_cache = True,
        cache_json_path = st.session_state['cdot_cost_data_config']['citation_cache_json_path'],
        cache_dir = st.session_state['cdot_cost_data_config']['citation_images_cache_dir'],
    )
    saq_citation_images, cost_data_citation_result = await asyncio.gather(
        saq_citation_task,
        cost_data_citation_task
    )
    st.session_state['citation_cache'][cache_key] = (saq_citation_images, cost_data_citation_result)
    return saq_citation_images, cost_data_citation_result

def clear_pdf_servers():
    if 'pdf_servers' in st.session_state:
        for server in st.session_state['pdf_servers'].values():
            server.destroy()
        del st.session_state['pdf_servers']

@st.dialog('View Source', width='large', on_dismiss = clear_pdf_servers)
def show_citation_dialog(
        item: ProjectItem,
        source_year: int,
        wab: float|None = None,
        cdot_name: str|None = None
):
    st.session_state['editor_key'] += 1  # force refresh of data editor
    st.markdown(f'### Finding source for: {item.item_name}')
    st.markdown(f'**Item Number:** {item.item_number} | **Unit:** {item.unit} | **Quantity:** {item.quantity} | **WAB:** \\$ {wab:,.2f}')


    if not st.session_state.get('saq_file_bytes') or not st.session_state.get('unstract_response_json'):
        st.error('Citation is only available when using "unstract whisper" file processing mode.')
        return

    start_time = time.time()
    with st.spinner('Generating citation (~30 seconds) . . .', show_time=True):
        saq_citation_images, (cost_data_citation_images, cost_data_citation_pages) = asyncio.run(get_estimate_citation_images(item, source_year, wab, cdot_name))
    end_time = time.time()

    if saq_citation_images or cost_data_citation_images:
        st.success(f'Found {len(saq_citation_images)+len(cost_data_citation_images)} citation image(s) in {end_time - start_time:.2f} seconds.')
        main_col, margin = st.columns([8,1])
        with main_col:
            if saq_citation_images:
                with st.expander('SAQ source(s)', expanded=True):
                    st.write('*scroll on images to zoom*')
                    for i, img in enumerate(saq_citation_images):
                        with st.container(key=f'aura-saq-{i}'):
                            image_zoom(img, mode = 'both', size = 1024, keep_resolution = True, zoom_factor = 4., increment = .3)
                        if i < len(saq_citation_images) - 1:
                            st.divider()
            else:
                st.markdown('No SAQ citations found!')
            if cost_data_citation_images:
                cost_item_pdf_path = st.session_state['cost_item_pdf_paths'][source_year]
                if 'pdf_servers' not in st.session_state:
                    st.session_state['pdf_servers'] = {}
                if source_year not in st.session_state['pdf_servers']:
                    st.session_state['pdf_servers'][source_year] = StreamlitPDFServer(cost_item_pdf_path)

                with st.expander('CDOT cost data book source(s)', expanded = True):
                    st.write('*scroll on images to zoom*')
                    for i, (img, page_no) in enumerate(zip(cost_data_citation_images, cost_data_citation_pages)):
                        with st.container(key=f'aura-cdot-{i}'):
                            image_zoom(img, mode='both', size=1024, keep_resolution=True, zoom_factor=4., increment=.3)
                        page_link = st.session_state['pdf_servers'][source_year].get_page_link(page=page_no)
                        st.markdown(f'[View full page in document (Page {page_no + 1}) :material/open_in_new:]({page_link})')
                        if i < len(cost_data_citation_images) - 1:
                            st.divider()
            else:
                st.markdown('No CDOT cost data citations found!')

        with margin:
            with st.container(height = 'stretch', width = 'stretch', key = 'gray-background'):
                st.write(' ')
                pass
    else:
        st.warning('No citations found for this item. The source may not be clearly identifiable in the document.')

def generate_estimate(
        saq_file,
        file_processing_mode: str
):
    # Clear citation cache when generating new estimate
    if 'citation_cache' in st.session_state:
        st.session_state['citation_cache'] = {}

    with (
        trace('Engineer\'s Estimate Workflow'),
        st.spinner('Estimating (may take several minutes) . . .', show_time=True),
        TemporaryDirectory() as tmpdir
    ):
        temp_table_path = os.path.join(tmpdir, saq_file.name)
        with open(temp_table_path, 'wb') as f:
            f.write(saq_file.getvalue())

        wab_partial_func = partial(get_wabs,
                                   claude_api_key=st.session_state['config']['anthropic_api_key'],
                                   openai_api_key=st.session_state['ddl_config']['openai_api_key'],
                                   openai_files_cache_path=st.session_state['ddl_config']['openai_files_cache_path'],
                                   openai_extract_project_items_prompt=st.session_state[
                                       'openai_extract_project_items_prompt'],
                                   claude_extract_project_items_prompt=st.session_state[
                                       'claude_extract_project_items_prompt'],
                                   search_agent_prompt=st.session_state['search_agent_prompt'],
                                   extract_wab_prompts=st.session_state['extract_wab_prompts'],
                                   cost_items_lists=st.session_state['cost_items_lists'],
                                   openai_model=st.session_state['config']['openai_model'],
                                   claude_model=st.session_state['config']['claude_model']
                                   )
        match file_processing_mode:
            case 'none':
                st.session_state['wab_items'], st.session_state['final_undecided_items'] = wab_partial_func(
                    project_items_table_path=temp_table_path)
            case 'unstract whisper':
                st.session_state['saq_file_bytes'] = saq_file.read()  # store PDF bytes for citation generation
                try:
                    st.session_state['unstract_response_json'] = whisper_pdf_text_extraction(
                        st.session_state['config']['unstract_api_key'], temp_table_path, return_json=True)
                except TimeoutError as e:
                    st.error(
                        'Unstract API request timed out. Please try again later, or try a different file processing mode.')
                    st.markdown(
                        'Unstract\'s LLMWhisperer API goes down occasionally due to high demand. check <https://status.unstract.com/> for current status.')
                    raise e
                project_items_table_ocr_str = st.session_state['unstract_response_json']['result_text']
                st.session_state['wab_items'], st.session_state['final_undecided_items'] = wab_partial_func(
                    project_items_table_path=temp_table_path, project_items_table_ocr_str=project_items_table_ocr_str)
            case 'google ocr':
                temp_ocr_table_path = os.path.join(tmpdir, 'ocr_' + saq_file.name)
                google_ocr_pdf_text_overlay(temp_table_path, temp_ocr_table_path, dpi=300)
                st.session_state['wab_items'], st.session_state['final_undecided_items'] = wab_partial_func(
                    project_items_table_path=temp_ocr_table_path)

        if not st.session_state['wab_items']:
            st.error(
                'Could not extract any project items from the table. Please check the file and try again, or try a different file processing mode.')


def render_estimate_table():

    #load the data into a dataframe and get excluded_indices
    wab_data = []
    excluded_indices = []
    for wab_index, item_dict in enumerate(st.session_state['wab_items']):
        item = item_dict['ProjectItem']
        year = item_dict['matched_year']
        wab = item_dict['wab_float']
        cdot_name = item_dict['cdot_name']
        if wab is None:
            excluded_indices.append(wab_index)
        else:
            wab_data.append([
                item.item_name,
                item.item_number,
                item.unit,
                item.quantity,
                year,
                wab,
                wab * item.quantity,
                cdot_name,
                wab_index,  # index within st.session_state['wab_items']
            ])
    st.divider()
    st.markdown('### Weighted Average Bids\n\n_click "Source" checkbox to view citation_')
    # st.markdown('_click "Source" checkbox to view citation_')
    wab_df = pd.DataFrame(wab_data, columns=['Item Name', 'Item Number', 'Unit', 'Quantity', 'Cost Data Book Year',
                                             'Weighted Average Bid', 'Calculated Cost', 'cdot_name', 'wab_index'])
    wab_df['Source'] = False
    wab_df['Delete'] = False

    citation_available = 'unstract_response_json' in st.session_state and st.session_state[
        'unstract_response_json'] is not None
    col_config = {
        'wab_index': None,  # hide index column
        'cdot_name': None,  # hide cdot_name column
        'Item Name': st.column_config.Column(width=200),
        'Item Number': st.column_config.Column(width=40),
        'Unit': st.column_config.Column(width=10),
        'Quantity': st.column_config.NumberColumn(width=30),
        'Cost Data Book Year': st.column_config.Column(width=80),
        'Weighted Average Bid': st.column_config.NumberColumn('Weighted Avg Bid', format='dollar', width=90),
        'Calculated Cost': st.column_config.NumberColumn('Calculated Cost', format='dollar', width=80),
        'Delete': st.column_config.CheckboxColumn('Delete', help='Check to remove this item from the estimate',
                                                  width=25),
        'Source': st.column_config.CheckboxColumn(disabled=not citation_available,
                                                  help='Check to show the source for this item from the estimate',
                                                  width=25)
    }
    if 'editor_key' not in st.session_state:
        st.session_state['editor_key'] = 0
    new_wab_df = st.data_editor(
        wab_df,
        disabled=['Item Name', 'Item Number', 'Unit', 'Cost Data Book Year'],
        # disable editing of all columns except WAB, quantity, and delete
        column_config=col_config,
        width='stretch',
        key=st.session_state['editor_key']
    )
    if not citation_available:
        st.info('Citation viewing is only available when using "unstract whisper" file processing mode.')

    # check for changes / citation requests / deletions
    for _, row in new_wab_df.iterrows():
        idx = row['wab_index']
        item_info_dict = st.session_state['wab_items'][idx]

        if row['Delete']:
            confirm_delete_dialog(item_info_dict['ProjectItem'].item_name, idx)
        if row['Source']:
            show_citation_dialog(
                item=item_info_dict['ProjectItem'],
                source_year=item_info_dict['matched_year'],
                wab=item_info_dict['wab_float'],
                cdot_name=item_info_dict['cdot_name']
            )
        if row['Weighted Average Bid'] != item_info_dict['wab_float']:
            item_info_dict['wab_float'] = row['Weighted Average Bid']
            st.session_state['wab_items'][idx] = item_info_dict
            st.rerun()
        if row['Quantity'] != item_info_dict['ProjectItem'].quantity:
            item_info_dict.quantity = row['Quantity']
            st.session_state['wab_items'][idx] = item_info_dict
            st.rerun()

    st.markdown(
        'You can edit the WAB values and quantities directly in the table above, get citations from the original document, or delete items using the checkbox.')
    total_estimate = sum(row['Calculated Cost'] for _, row in wab_df.iterrows())

    if st.button('Add Item Manually'):
        add_item_dialog()
    if (not excluded_indices) and (not st.session_state['final_undecided_items']):
        st.markdown(f'### Total Engineers Estimate: :green[${total_estimate:,.2f}]')
        st.success('Estimation complete!')
    else:
        st.markdown(f'### Total Engineers Estimate: :red[${total_estimate:,.2f}]')
        st.error(
            f'Some items were excluded from the estimate due to missing WABs, or due to parsing errors. Please review the excluded items below to fill in the missing values. Note that some items may be duplicates or not present in the provided table, and thus should not be included')
        st.markdown('### Excluded Items')
        for i in excluded_indices:
            item_info_dict = st.session_state['wab_items'][i]
            item = item_info_dict['ProjectItem']
            year = item_info_dict['matched_year']
            wab = item_info_dict['wab_float']
            with st.expander(item.item_name):
                st.markdown(f'- **Item Number:** {item.item_number}')
                st.markdown(f'- **Unit:** {item.unit}')
                st.markdown(f'- **Quantity:** {item.quantity}')
                st.markdown(
                    f'The WAB could not be determined for this item. Enter it below to include it in the estimate.')
                user_wab = st.number_input(f'Enter WAB for item {item.item_name} (per unit)', width=200)
                if st.button(f'Add {item.item_name} to Estimate with new WAB', key=f'add_{i}'):
                    item_info_dict['wab_float'] = user_wab
                    st.session_state['wab_items'][i] = item_info_dict
                    st.rerun()
        for i, item_dict in enumerate(st.session_state['final_undecided_items']):
            with st.expander(item_dict.get('item_name', 'Unnamed Item')):
                if 'item_name' in item_dict:
                    item_name = item_dict['item_name']
                else:
                    item_name = st.text_input(f'Item name', value=item_dict.get('item_name', ''),
                                              key=f'undecided_item_name_{i}')
                item_number = st.text_input(f'Item number', value=item_dict.get('item_number', ''),
                                            key=f'undecided_item_number_{i}')
                item_unit = st.text_input(f'Unit', value=item_dict.get('unit', ''), key=f'undecided_item_unit_{i}')
                item_quantity = st.number_input(f'Quantity', value=item_dict.get('quantity', 0),
                                                key=f'undecided_item_quantity_{i}')
                user_wab = st.number_input(
                    f'Enter WAB for item {item_dict.get('item_name', 'Unnamed Item')} (per unit)', width=200,
                    key=f'undecided_item_wab_{i}')

                st.markdown(
                    f'This item could not be confidently parsed from the table. Verify it is not already included before adding to the estimate.')
                addable = (item_name != '') and (item_number != '') and (item_unit != '') and (user_wab > 0)
                if st.button(f'Add "{item_name}" to Estimate', key=f'add_undecided_{i}', disabled=(not addable)):
                    new_item = ProjectItem(
                        item_name=item_name,
                        item_number=item_number,
                        unit=item_unit,
                        quantity=float(item_dict['quantity'])  # default to 1.0 if not provided
                    )
                    new_item_info_dict = {
                        'cdot_name': None,
                        'wab_float': user_wab,
                        'matched_year': None,
                        'ProjectItem': new_item
                    }
                    st.session_state['wab_items'].append(new_item_info_dict)
                    st.session_state['final_undecided_items'].pop(i)
                    st.rerun()
                if st.button('Delete Item', key=f'delete_undecided_{i}'):
                    st.session_state['final_undecided_items'].pop(i)
                    st.rerun()

# <editor-fold> Config Loading
if 'cdot_cost_data_config' not in st.session_state:
    with open('../cdotcostdata/config.toml', 'rb') as f:
        st.session_state['cdot_cost_data_config'] = tomli.load(f)
    with open(st.session_state['cdot_cost_data_config']['cost_data_citation_prompt_path'], 'r', encoding='utf-8') as f:
        st.session_state['cost_data_citation_prompt'] = f.read()



if 'ddl_config' not in st.session_state:
    with open('../proposalwriting/ddl_config.toml', 'rb') as f:
        st.session_state['ddl_config'] = tomli.load(f)

if 'config' not in st.session_state:
    with open('config.toml', 'rb') as f:
        st.session_state['config'] = tomli.load(f)
    with open(st.session_state['config']['openai_extract_project_items_prompt_path'], 'r', encoding='utf-8') as f:
        st.session_state['openai_extract_project_items_prompt'] = f.read()
    with open(st.session_state['config']['claude_extract_project_items_prompt_path'], 'r', encoding='utf-8') as f:
        st.session_state['claude_extract_project_items_prompt'] = f.read()
    with open(st.session_state['config']['search_agent_prompt_path'], 'r', encoding='utf-8') as f:
        st.session_state['search_agent_prompt'] = f.read()
    with open(st.session_state['config']['citation_prompt_path'], 'r', encoding='utf-8') as f:
        st.session_state['citation_prompt'] = f.read()
    st.session_state['extract_wab_prompts'] = {}
    for wab_prompt_ref in st.session_state['config']['extract_wab_prompts_paths']:
        with open(wab_prompt_ref['path'], 'r', encoding='utf-8') as f:
            st.session_state['extract_wab_prompts'][wab_prompt_ref['year']] = f.read()
# </editor-fold>


if 'cost_items_lists' not in st.session_state:
    st.session_state['cost_items_lists'] = []
    st.session_state['cost_item_pdf_paths'] = {}
    for cost_book_ref in st.session_state['cdot_cost_data_config']['cost_data_books']:
        with open(cost_book_ref['cost_items_pkl_path'], 'rb') as f:
            cost_items = pickle.load(f)
        st.session_state['cost_items_lists'].append([cost_items, cost_book_ref['year']])

        st.session_state['cost_item_pdf_paths'][cost_book_ref['year']] = cost_book_ref['pdf_path']
    st.session_state['cost_items_lists'] = sorted(st.session_state['cost_items_lists'], key=lambda x: x[1], reverse=True)  # sort by year, most recent first
    st.session_state['cost_items_lists'] = {year: items for items, year in st.session_state['cost_items_lists']}  # convert to dict for easy access


# <editor-fold> Page Styling
gray_background_css = '''
<style>
.st-key-gray-background {
    background-color: #eaeaea;
}
</style>
'''
st.html(gray_background_css)

BACKGROUND_IMAGE_PATH = st.secrets.get("app_background_image_path")
mime_type, _ = mimetypes.guess_type(BACKGROUND_IMAGE_PATH)
with open(BACKGROUND_IMAGE_PATH, "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.set_page_config(page_title = 'Engineer\'s Estimate', page_icon = '📊', layout = 'wide')

background_image_css = f"""
<style>
.stApp {{
    background-image: url("data:{mime_type};base64,{encoded}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.html(background_image_css)

# background for main central container
translucent_bg_element="""
<style>
.st-key-translucent{
    background-color: rgba(220,220,220,.9);
    background-size: cover;
    padding: 25px;
}"""
st.html(translucent_bg_element)

# glowing aura for citations
aura_css = """
<style>
[class*="st-key-aura"] {
    transition: box-shadow 0.3s ease;
}

[class*="st-key-aura"]:hover {
    box-shadow: 0 0 10px rgba(0, 168, 232, 0.6),
                0 0 20px rgba(0, 168, 232, 0.4),
                0 0 30px rgba(0, 168, 232, 0.2) !important;
    animation: pulse-glow 2s ease-in-out infinite !important;
}

@keyframes pulse-glow {
    0%, 100% {
        box-shadow: 0 0 10px rgba(0, 168, 232, 0.6),
                    0 0 20px rgba(0, 168, 232, 0.4),
                    0 0 30px rgba(0, 168, 232, 0.2) !important;
    }
    50% {
        box-shadow: 0 0 15px rgba(0, 168, 232, 0.8),
                    0 0 30px rgba(0, 168, 232, 0.6),
                    0 0 45px rgba(0, 168, 232, 0.4) !important;
    }
}
</style>
"""
st.html(aura_css)

# </editor-fold>


l_bg, center, r_bg = st.columns([1, 10, 1], gap='large')
main_container = center.container(key='translucent') # central container, for content
l_margin, main_col, r_margin = main_container.columns([1, 10, 1])

with main_col:
    st.title('Engineer\'s Estimate')
    st.markdown('_AI generated results are not guaranteed to be accurate._  \n**Uploaded files will be sent to OpenAI and Anthropic via API. Their policies (as of 11/18/25) are not to use this data to train their models. Check the [OpenAI:material/open_in_new:](https://platform.openai.com/docs/guides/your-data) and [Anthropic:material/open_in_new:](https://privacy.claude.com/en/collections/10663361-commercial-customers) privacy pages for the most recent privacy information.**')

    saq_file = st.file_uploader("Upload project items table (PDF only)", accept_multiple_files=False, width = 500, type = 'pdf')

    file_processing_mode = st.pills(
        label = 'file processing mode ("unstract whisper" recommended)', options=['unstract whisper', 'google ocr', 'none'], default = 'unstract whisper', selection_mode = 'single')

    if st.button('Get Engineers Estimate', disabled = (saq_file is None)):
        generate_estimate(saq_file, file_processing_mode)
        st.rerun()
    if 'wab_items' in st.session_state:
        render_estimate_table()
st.container(height = 400, border = False) # if you want to be able to scroll to see the background