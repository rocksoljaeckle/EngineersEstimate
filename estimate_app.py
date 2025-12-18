from tempfile import TemporaryDirectory
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
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
from pandas import DataFrame
import base64


from estimate_utils import (
    get_project_items,
    Estimator,
)
from estimate_global_utils import (
    get_unstract_citation_images,
    StreamlitPDFServer,
)
from estimate_cdot_utils import (
    ProjectItem,
    extract_item_citation_images_pages,
)



# <editor-fold> Authentication

if not st.user.is_logged_in:
    if st.button('Log in with Microsoft'):
        st.login('microsoft')
    st.stop()
# </editor-fold>

def get_aggrid_options(df: DataFrame, hidden_cols: list[str]):
    """Get AgGrid grid options for a given DataFrame."""
    gb = GridOptionsBuilder.from_dataframe(dataframe=df)
    gb.configure_selection('single', use_checkbox=False)
    gb.configure_auto_height(autoHeight=False)
    for hidden in hidden_cols:
        gb.configure_column(hidden, hide = True)
    return gb.build()

@st.dialog('Help', width='large')
def show_help_dialog():
    """Display the help documentation in a dialog."""
    help_html_path = os.path.join(os.path.dirname(__file__), 'help.html')
    if os.path.exists(help_html_path):
        with open(help_html_path, 'r', encoding='utf-8') as f:
            help_html = f.read()
        st.components.v1.html(help_html, height=700, scrolling=True)
    else:
        st.error('Help file not found.')

@st.dialog('Confirm rerun', on_dismiss='rerun')
def confirm_rerun_dialog():
    st.markdown('Are you sure you want to rerun the estimation? This will clear all results.')
    if st.button('Rerun', type = 'primary'):
        st.session_state['saq_file_name'] = saq_file.name
        generate_estimate(saq_file)
        st.rerun()
    if st.button('Cancel'):
        st.rerun()

@st.dialog('Confirm Delete', on_dismiss = 'rerun')
def confirm_delete_dialog(item_name: str, item_index: int):
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
    if st.button(f'Add "{item_name}" to Estimate', key=f'add_undecided', disabled=(not addable)):
        new_item = ProjectItem(
            item_name=item_name,
            item_number=item_number,
            unit=item_unit,
            quantity=item_quantity
        )
        new_item_dict = {
            'cdot_name': None,
            'wab_float': user_wab,
            'matched_year': None,
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

    openai_client = AsyncOpenAI(api_key=st.secrets['openai_api_key'])
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
        unstract_api_key=st.secrets['unstract_api_key'],
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
        saq_file
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
        st.session_state['saq_file_bytes'] = saq_file.getvalue()
        async_openai_client = AsyncOpenAI(api_key = st.secrets['openai_api_key'])
        async_claude_client = AsyncAnthropic(api_key=st.secrets['anthropic_api_key'])
        estimator = Estimator(
            project_items_table_path = temp_table_path,
            unstract_api_key = st.secrets['unstract_api_key'],
            openai_api_key = st.secrets['openai_api_key'],
            async_openai_client=async_openai_client,
            openai_files_cache_path=st.session_state['global_config']['openai_files_cache_path'],
            openai_extract_project_items_prompt=st.session_state['openai_extract_project_items_prompt'],
            openai_model = st.session_state['config']['openai_model'],
            async_claude_client=async_claude_client,
            claude_extract_project_items_prompt=st.session_state['claude_extract_project_items_prompt'],
            claude_model = st.session_state['config']['claude_model'],
            cost_items_lists = st.session_state['cost_items_lists'],
            search_agent_prompt = st.session_state['search_agent_prompt'],
            extract_wab_prompts = st.session_state['extract_wab_prompts'],
            wab_cache_path = st.session_state['cdot_cost_data_config']['wab_cache_json_path'],
        )
        st.session_state['wab_items'], st.session_state['final_undecided_items'] = asyncio.run(estimator.get_wabs())
        st.session_state['unstract_response_json'] = estimator.table_unstract_json_response

        if not st.session_state['wab_items']:
            st.error(
                'Could not extract any project items from the table. Please check the file and try again, or try a different file processing mode.')


def get_estimate_table_html(
        source_file_name: str,
        data_df: pd.DataFrame,
        hidden_cols: list[str],
        total_estimate: float,
        excluded_indices: list[int]
) -> str:
    missing_items = len(excluded_indices) + len(st.session_state['final_undecided_items']) > 0
    display_columns = [col for col in data_df.columns if col not in hidden_cols]

    table_html = f'<html><head><title>Estimate items from "{source_file_name}"</title></head>'
    table_html += f'<body><h1>Cost items table from "{source_file_name}"</h1>'
    table_html += f'<h3>Total Engineer\'s Estimate: ${total_estimate:,.2f}</h3>'
    if missing_items:
        table_html += '<strong style="color:red;">Note: Some items were excluded from this estimate due to missing WABs or parsing errors.</strong>'
    table_html += '<div style="height: 20px;"></div>'
    table_html += data_df.to_html(columns = display_columns, index=True, justify='center')
    table_html += '<p>The following items were excluded from the estimate and table above:</p>'
    table_html += '<ul>'
    for i in excluded_indices:
        item_name = st.session_state['wab_items'][i]['ProjectItem'].item_name
        table_html += f'<li>{item_name}</li>'
    for item_dict in st.session_state['final_undecided_items']:
        item_name = item_dict.get('item_name', 'Unnamed Item')
        table_html += f'<li>{item_name}</li>'
    table_html += '</ul></body></html>'

    return table_html


def render_estimate_table():
    st.divider()
    st.markdown('### Weighted Average Bids')
    st.markdown('_click a row to edit, delete, or view source for an item (shown below table)_')

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

    wab_df = pd.DataFrame(wab_data, columns=['Item Name', 'Item Number', 'Unit', 'Quantity', 'Cost Data Book Year',
                                             'Weighted Average Bid', 'Calculated Cost', 'cdot_name', 'wab_index'])
    total_estimate = sum(row['Calculated Cost'] for _, row in wab_df.iterrows())
    table_html = get_estimate_table_html(
        source_file_name = st.session_state['saq_file_name'],
        data_df = wab_df,
        hidden_cols = ['cdot_name', 'wab_index'],
        total_estimate = total_estimate,
        excluded_indices = excluded_indices
    )
    st.download_button('Download Estimate Table as HTML', data=table_html, file_name=f'estimate_table_{st.session_state["saq_file_name"]}.html', mime='text/html')

    grid_options = get_aggrid_options(wab_df, hidden_cols=['cdot_name', 'wab_index'])
    wab_grid = AgGrid(
        wab_df,
        gridOptions = grid_options,
        update_mode = 'SELECTION_CHANGED',
        height = 500
    )
    selected = wab_grid['selected_rows']

    if selected is not None and len(selected) > 0:
        selected_row = selected.iloc[0]
        idx = selected_row['wab_index']
        item_info_dict = st.session_state['wab_items'][idx]
        item = item_info_dict['ProjectItem']
        wab = item_info_dict['wab_float']

        with st.container(border=True):
            st.markdown(f'### Selected Item : "{item.item_name}"')
            st.markdown(
                f'**Item Number:** {item.item_number} | **Unit:** {item.unit} | **Quantity:** {item.quantity} | **WAB:** \\$ {wab:,.2f}')

            l_col, mid_col, r_col = st.columns([1,1,3])
            with l_col.container(border=True, height = 200):
                new_wab = st.number_input('Edit Weighted Average Bid', value=selected_row['Weighted Average Bid'], width=200)
                if st.button('Update WAB for selected item'):
                    item_info_dict['wab_float'] = new_wab
                    st.session_state['wab_items'][idx] = item_info_dict
                    st.rerun()

            with mid_col.container(border=True, height = 200):
                # st.write('')
                new_quantity = st.number_input('Edit Quantity', value=selected_row['Quantity'], width=200)
                if st.button('Update quantity for selected item'):
                    item_info_dict['ProjectItem'].quantity = new_quantity
                    st.session_state['wab_items'][idx] = item_info_dict
                    st.rerun()

            l_col2, r_col2 = st.columns([1,4])
            if l_col2.button('View Source for Selected Item'):
                show_citation_dialog(
                    item=item,
                    source_year=item_info_dict['matched_year'],
                    wab=item_info_dict['wab_float'],
                    cdot_name=item_info_dict['cdot_name']
                )
            if r_col2.button('Delete Selected Item'):
                confirm_delete_dialog(item.item_name, idx)

    st.markdown('_click a row to edit, delete, or view source for an item (shown below table)_')


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



if 'global_config' not in st.session_state:
    with open('../GlobalUtils/config.toml', 'rb') as f:
        st.session_state['global_config'] = tomli.load(f)

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
    # st.session_state['cost_items_lists'] = sorted(st.session_state['cost_items_lists'], key=lambda x: x[1], reverse=True)  # sort by year, most recent first
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

BACKGROUND_IMAGE_PATH = st.session_state['config']['app_background_image_path']
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
}
</style>"""
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
    title_col, help_col = st.columns([8, 1])
    with title_col:
        st.title('Engineer\'s Estimate')
    with help_col:
        st.write('')  # spacing
        if st.button(':material/help: Help', type='tertiary', use_container_width=True):
            show_help_dialog()
    st.markdown('_AI generated results are not guaranteed to be accurate._  \n**Uploaded files will be sent to OpenAI and Anthropic via API. Their policies (as of 11/18/25) are not to use this data to train their models. Check the [OpenAI:material/open_in_new:](https://platform.openai.com/docs/guides/your-data) and [Anthropic:material/open_in_new:](https://privacy.claude.com/en/collections/10663361-commercial-customers) privacy pages for the most recent privacy information.**')

    saq_file = st.file_uploader("Upload project items table (PDF only)", accept_multiple_files=False, width = 500, type = 'pdf')

    if st.button('Get Engineers Estimate', disabled = (saq_file is None)):
        if 'wab_items' in st.session_state:
            confirm_rerun_dialog()
        else:
            st.session_state['saq_file_name'] = saq_file.name
            generate_estimate(saq_file)
            st.rerun()
    if 'wab_items' in st.session_state:
        render_estimate_table()
st.container(height = 400, border = False) # if you want to be able to scroll to see the background