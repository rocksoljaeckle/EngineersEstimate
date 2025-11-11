from tempfile import TemporaryDirectory
import pandas as pd
import streamlit as st
import tomli
import asyncio
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from agents import set_default_openai_key, trace
import os
import pickle
import sys
from plyer import notification
import datetime
from functools import partial

from estimate_utils import (
    get_project_items,
    batch_item_weighted_avg_bids,
    ProjectItem,
    ProjectItemsList
)
from GlobalUtils.ocr import google_ocr_pdf_text_overlay, whisper_pdf_text_extraction
from GlobalUtils.unstract_citation import get_citation_images
from CDOTCostData.cost_data_utils import ItemSearchWrapper


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

st.set_page_config(page_title = 'Engineer\'s Estimate', page_icon = '📊', layout = 'wide')
st.title('Engineer\'s Estimate')
st.markdown('_AI generated results are not guaranteed to be accurate._')
st.markdown('**Uploaded files will be sent to OpenAI via API. OpenAI\'s policy (as of 10/21/25) is not to use this data to train their models. See [this page](https://platform.openai.com/docs/guides/your-data) for the most recent privacy information.**')

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

    Returns a list of tuples (item, year, wab).'''
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

    wabs_n_years = asyncio.run(
        batch_item_weighted_avg_bids(project_items_list, item_search_wrappers, search_agent_prompt,
                                     extract_wab_prompts, openai_async_client, model=openai_model)
    )

    out_items = [(item, year, wab) for item, (wab, year) in zip(project_items_list.items, wabs_n_years)]
    return out_items, final_undecided_items

@st.dialog('Confirm Delete', on_dismiss = 'rerun')
def confirm_delete_dialog(item_name: str, item_index: int):
    st.markdown(f'Are you sure you want to delete item **{item_name}** from the estimate? This action cannot be undone.')
    if st.button('Confirm Delete'):
        st.session_state['wab_items'].pop(item_index)
        st.session_state['editor_key'] += 1  # force refresh of data editor
        st.rerun()
    if st.button('Cancel'):
        st.session_state['editor_key'] += 1  # force refresh of data editor
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
        st.session_state['wab_items'].append((new_item, None, user_wab))
        st.rerun()

if 'cdot_cost_data_config' not in st.session_state:
    with open('../cdotcostdata/config.toml', 'rb') as f:
        st.session_state['cdot_cost_data_config'] = tomli.load(f)
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
    st.session_state['extract_wab_prompts'] = {}
    for wab_prompt_ref in st.session_state['config']['extract_wab_prompts_paths']:
        with open(wab_prompt_ref['path'], 'r', encoding='utf-8') as f:
            st.session_state['extract_wab_prompts'][wab_prompt_ref['year']] = f.read()

if 'cost_items_lists' not in st.session_state:
    st.session_state['cost_items_lists'] = []
    for cost_book_ref in st.session_state['cdot_cost_data_config']['cost_data_books']:
        with open(cost_book_ref['cost_items_pkl_path'], 'rb') as f:
            cost_items = pickle.load(f)
        st.session_state['cost_items_lists'].append([cost_items, cost_book_ref['year']])
    st.session_state['cost_items_lists'] = sorted(st.session_state['cost_items_lists'], key=lambda x: x[1], reverse=True)  # sort by year, most recent first
    st.session_state['cost_items_lists'] = {year: items for items, year in st.session_state['cost_items_lists']}  # convert to dict for easy access

file = st.file_uploader("Upload project items table (PDF only)", accept_multiple_files=False, width = 500, type = 'pdf')

file_processing_mode = st.pills(
    label = 'file processing mode ("unstract whisper" recommended)', options=['none', 'google ocr', 'unstract whisper'], default = 'unstract whisper', selection_mode = 'single')


if st.button('Get Engineers Estimate', disabled = (file is None)):
    with (
            trace('Engineer\'s Estimate Workflow'),
            st.spinner('Estimating (may take several minutes) . . .', show_time = True),
            TemporaryDirectory() as tmpdir
    ):
        temp_table_path = os.path.join(tmpdir, file.name)
        with open(temp_table_path, 'wb') as f:
            f.write(file.read())

        wab_partial_func = partial(get_wabs,
            claude_api_key=st.session_state['config']['anthropic_api_key'],
            openai_api_key=st.session_state['ddl_config']['openai_api_key'],
            openai_files_cache_path=st.session_state['ddl_config']['openai_files_cache_path'],
            openai_extract_project_items_prompt=st.session_state['openai_extract_project_items_prompt'],
            claude_extract_project_items_prompt=st.session_state['claude_extract_project_items_prompt'],
            search_agent_prompt=st.session_state['search_agent_prompt'],
            extract_wab_prompts=st.session_state['extract_wab_prompts'],
            cost_items_lists=st.session_state['cost_items_lists'],
            openai_model = st.session_state['config']['openai_model'],
            claude_model = st.session_state['config']['claude_model']
        )
        match file_processing_mode:
            case 'none':
                st.session_state['wab_items'], st.session_state['final_undecided_items'] = wab_partial_func(project_items_table_path=temp_table_path)
            case 'unstract whisper':
                project_items_table_ocr_str = whisper_pdf_text_extraction(st.session_state['config']['unstract_api_key'], temp_table_path)
                st.session_state['wab_items'], st.session_state['final_undecided_items'] = wab_partial_func(project_items_table_path=temp_table_path, project_items_table_ocr_str=project_items_table_ocr_str)
            case 'google ocr':
                temp_ocr_table_path = os.path.join(tmpdir, 'ocr_' + file.name)
                google_ocr_pdf_text_overlay(temp_table_path, temp_ocr_table_path, dpi=300)
                st.session_state['wab_items'], st.session_state['final_undecided_items'] = wab_partial_func(project_items_table_path=temp_ocr_table_path)
        if not st.session_state['wab_items']:
            st.error('Could not extract any project items from the table. Please check the file and try again, or try a different file processing mode.')

if 'wab_items' in st.session_state:
    wab_data = []
    excluded_indices = []
    for i, (item, year, wab) in enumerate(st.session_state['wab_items']):
        if wab is None:
            excluded_indices.append(i)
        else:
            wab_data.append([
                item.item_name,
                item.item_number,
                item.unit,
                item.quantity,
                year,
                wab,
                wab * item.quantity,
                i # index within st.session_state['wab_items']
            ])

    st.markdown('### Weighted Average Bids')
    wab_df = pd.DataFrame(wab_data, columns=['Item Name', 'Item Number', 'Unit', 'Quantity', 'Cost Data Book Year', 'Weighted Average Bid', 'Calculated Cost','index'])
    wab_df['Delete'] = False
    col_config = {
        'index': None, # hide index column
        'Weighted Average Bid': st.column_config.NumberColumn('Weighted Average Bid', format = 'dollar', width = 25),
        'Calculated Cost': st.column_config.NumberColumn('Calculated Cost', format = 'dollar', width = 30),
        'Delete': st.column_config.CheckboxColumn('Delete', help = 'Check to remove this item from the estimate', width = 5)
    }
    if 'editor_key' not in st.session_state:
        st.session_state['editor_key'] = 0
    new_wab_df = st.data_editor(
        wab_df,
        disabled = ['Item Name', 'Item Number', 'Unit', 'Cost Data Book Year'],  # disable editing of all columns except WAB, quantity, and delete
        column_config =col_config,
        width='stretch',
        key = st.session_state['editor_key']
    )
    for _, row in new_wab_df.iterrows():
        idx = row['index']
        if row['Delete']:
            confirm_delete_dialog(st.session_state['wab_items'][idx][0].item_name, idx)
        if row['Weighted Average Bid'] != st.session_state['wab_items'][idx][2]:
            st.session_state['wab_items'][idx] = (
                st.session_state['wab_items'][idx][0], # item
                st.session_state['wab_items'][idx][1], # year
                row['Weighted Average Bid'] # updated wab
            )
            st.rerun()
        if row['Quantity'] != st.session_state['wab_items'][idx][0].quantity:
            new_item = st.session_state['wab_items'][idx][0]
            new_item.quantity = row['Quantity']
            st.session_state['wab_items'][idx] = (
                new_item,
                st.session_state['wab_items'][idx][1], # year
                st.session_state['wab_items'][idx][2] # wab
            )
            st.rerun()


    st.markdown('You can edit the WAB values and quantities directly in the table above, or delete items using the checkbox.')
    total_estimate = sum(row['Calculated Cost'] for _, row in wab_df.iterrows())

    if st.button('Add Item Manually'):
        add_item_dialog()
    if (not excluded_indices) and (not st.session_state['final_undecided_items']):
        st.markdown(f'### Total Engineers Estimate: :green[${total_estimate:,.2f}]')
        st.success('Estimation complete!')
    else:
        st.markdown(f'### Total Engineers Estimate: :red[${total_estimate:,.2f}]')
        st.error(f'Some items were excluded from the estimate due to missing WABs, or due to parsing errors. Please review the excluded items below to fill in the missing values. Note that some items may be duplicates or not present in the provided table, and thus should not be included')
        st.markdown('### Excluded Items')
        for i in excluded_indices:
            item, year, wab = st.session_state['wab_items'][i]
            with st.expander(item.item_name):
                st.markdown(f'- **Item Number:** {item.item_number}')
                st.markdown(f'- **Unit:** {item.unit}')
                st.markdown(f'- **Quantity:** {item.quantity}')
                st.markdown(f'The WAB could not be determined for this item. Enter it below to include it in the estimate.')
                user_wab = st.number_input(f'Enter WAB for item {item.item_name} (per unit)', width = 200)
                if st.button(f'Add {item.item_name} to Estimate with new WAB', key=f'add_{i}'):
                    st.session_state['wab_items'][i] = (item, year, user_wab)
                    st.rerun()
        for i, item_dict in enumerate(st.session_state['final_undecided_items']):
            with st.expander(item_dict.get('item_name', 'Unnamed Item')):
                if 'item_name' in item_dict:
                    item_name = item_dict['item_name']
                else:
                    item_name = st.text_input(f'Item name', value = item_dict.get('item_name', ''), key=f'undecided_item_name_{i}')
                item_number = st.text_input(f'Item number', value = item_dict.get('item_number', ''), key=f'undecided_item_number_{i}')
                item_unit = st.text_input(f'Unit', value = item_dict.get('unit', ''), key=f'undecided_item_unit_{i}')
                item_quantity = st.number_input(f'Quantity', value = item_dict.get('quantity', 0), key=f'undecided_item_quantity_{i}')
                user_wab = st.number_input(f'Enter WAB for item {item_dict.get('item_name', 'Unnamed Item')} (per unit)', width = 200, key=f'undecided_item_wab_{i}')

                st.markdown(f'This item could not be confidently parsed from the table. Verify it is not already included before adding to the estimate.')
                addable = (item_name != '') and (item_number != '') and (item_unit != '') and (user_wab > 0)
                if st.button(f'Add "{item_name}" to Estimate', key=f'add_undecided_{i}', disabled = (not addable)):
                    new_item = ProjectItem(
                        item_name = item_name,
                        item_number = item_number,
                        unit = item_unit,
                        quantity = float(item_dict['quantity']) # default to 1.0 if not provided
                    )
                    st.session_state['wab_items'].append((new_item, None, user_wab))
                    st.session_state['final_undecided_items'].pop(i)
                    st.rerun()
                if st.button('Delete Item', key=f'delete_undecided_{i}'):
                    st.session_state['final_undecided_items'].pop(i)
                    st.rerun()