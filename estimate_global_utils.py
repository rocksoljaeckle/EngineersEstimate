import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
import fitz
import httpx
from openai import AsyncOpenAI
import io
from PIL import Image, ImageDraw
from pydantic import BaseModel

class CitationLines(BaseModel):
    lines: list[int]

# globalutils.ocr
async def async_whisper_pdf_text_extraction(
        unstract_api_key: str,
        input_pdf_path: str,
        retry_wait_step=1.,
        max_retry_time=30.,
        wait_step=1.,
        max_wait_time=30.,
        return_json=False,
        add_line_nos = False
):
    creation_start_time = time.time()

    with open(input_pdf_path, 'rb') as pdf_file:
        pdf_data = pdf_file.read()

    BASE_URL = 'https://llmwhisperer-api.us-central.unstract.com/api/v2'
    auth_headers = {'unstract-key': unstract_api_key}
    create_params = {}
    if add_line_nos:
        create_params['add_line_nos'] = True


    async with httpx.AsyncClient() as client:
        # Retry loop for job creation with rate limit handling
        while time.time() - creation_start_time < max_retry_time:
            create_job_response = await client.post(
                f'{BASE_URL}/whisper',
                headers=auth_headers,
                params=create_params,
                content=pdf_data
            )

            if create_job_response.status_code == 429:
                print(f"Rate limited, retrying in {retry_wait_step} seconds...")
                await asyncio.sleep(retry_wait_step)
            else:
                create_job_response.raise_for_status()
                whisper_hash = create_job_response.json()['whisper_hash']
                break
        else:
            raise TimeoutError(f"Could not create Whisper job within {max_retry_time} seconds due to rate limiting")

        # Status polling loop
        status_start_time = time.time()
        complete = False

        while not complete:
            status_response = await client.get(
                f'{BASE_URL}/whisper-status',
                headers=auth_headers,
                params={'whisper_hash': whisper_hash}
            )

            if status_response.json()['status'] == 'error':
                raise RuntimeError(f"Whisper job failed: {status_response.json()}")
            elif status_response.json()['status'] == 'processed':
                complete = True
            elif time.time() - status_start_time > max_wait_time:
                raise TimeoutError(f"Whisper job did not complete within {max_wait_time} seconds")
            else:
                await asyncio.sleep(wait_step)

        # Retrieve results
        result_response = await client.get(
            f'{BASE_URL}/whisper-retrieve',
            headers=auth_headers,
            params={'whisper_hash': whisper_hash}
        )

        if return_json:
            return result_response.json()

        return result_response.json()['result_text']

#globalutils.citation
def render_pdf_page_with_highlights(
        pdf_source: str | bytes,
        page: int,
        bboxes: list[list[float]],
        highlight_color: tuple[float, float, float] = (1, 1, 0),  # RGB 0-1, default yellow
        highlight_opacity: float = 0.3,
        zoom: float = 2.0  # Higher = better quality, 2.0 is good default
) -> Image.Image:
    """
    Render a PDF page with a highlighted bounding box.

    Args:
        pdf_source: Path to PDF file, or PDF bytes
        page: Page number (0-indexed)
        bbox: Bounding box as [x0, y0, x1, y1] in PDF coordinates
        highlight_color: RGB tuple with values 0-1
        highlight_opacity: Transparency of highlight (0-1)
        zoom: Rendering resolution multiplier

    Returns:
        PIL Image with highlighted region
    """
    if isinstance(pdf_source, bytes):
        doc = fitz.open(stream=pdf_source)
    elif isinstance(pdf_source, str):
        doc = fitz.open(pdf_source)
    else:
        raise ValueError('pdf_source must be str path or bytes')

    if page < 0 or page >= len(doc):
        raise ValueError(f"Page {page} out of range. PDF has {len(doc)} pages.")

    pdf_page = doc[page]

    # Render page at higher resolution
    mat = fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat)

    # Convert to PIL Image
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    # Create a semi-transparent overlay
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw highlight rectangles
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        scaled_bbox = [x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom]


        # Convert RGB 0-1 to 0-255
        color_255 = tuple(int(c * 255) for c in highlight_color)
        alpha = int(highlight_opacity * 255)
        fill_color = (*color_255, alpha)

        # Draw filled rectangle
        draw.rectangle(scaled_bbox, fill=fill_color, width=3)

    # Composite the overlay onto the original image
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)

    doc.close()

    return img.convert('RGB')  # Convert back to RGB for st.image

#globalutils.citation
async def find_best_openai_lines(
        text: str,
        query: str,
        citation_prompt: str,
        openai_client: AsyncOpenAI,
        openai_model: str = 'gpt-5'
):
    marked_lines =  ''
    for line_ind, line in enumerate(text.splitlines(keepends=True)):
        marked_lines += f'<<LINE {line_ind}>> {line}'
    lineate_input = [
        {
            'role': 'system',
            'content': citation_prompt
        },
        {
            'role': 'user',
            'content': f'Document:\n{marked_lines}\n\nQuery: "{query}"'
        }
    ]
    response = await openai_client.responses.parse(
        model=openai_model,
        input=lineate_input,
        text_format = CitationLines
    )
    lines = response.output_parsed.lines
    return lines

#globalutils.citation
async def find_citation_bboxes_normed(
        unstract_response_json: dict,
        citation_query: str,
        citation_prompt: str,
        openai_client: AsyncOpenAI
):
    """
    Find citation bounding boxes in normalized coordinates (0-1) for a given citation query.
    Args:
        unstract_response_json: JSON response from Unstract OCR
        citation_query: Citation text to search for
        openai_client: AsyncOpenAI client instance
    Returns:
        List of dicts with 'page' and 'bbox' keys in normalized coordinates (0-1), or None if not found
        """
    document_text = unstract_response_json['result_text']
    document_text_no_page_breaks = ''
    for line in document_text.splitlines(keepends=True):
        if line.strip() != '<<<':
            document_text_no_page_breaks += line
    lines = await find_best_openai_lines(
        document_text_no_page_breaks,
        citation_query,
        citation_prompt,
        openai_client
    )
    if lines is None:
        return None

    line_whisper_boxes = [unstract_response_json['line_metadata'][line_ind] for line_ind in lines]

    line_boxes = [
        {
            'page': line_box[0],
            'bbox': [0.01, (line_box[1]-line_box[2])/line_box[3], 0.99, (line_box[1])/line_box[3]]
        }
        for line_box in line_whisper_boxes
    ]
    return line_boxes # normed to (0,1)

#globalutils.citation
def render_pdf_bboxes_to_images(
        citation_bboxes: list[dict],
        pdf_source: str | bytes,
):
    """
    Generate images for each page in source document, with  citation bounding boxes highlighted.

    Args:
        citation_bboxes: List of dicts with 'page' and 'bbox' keys in normalized coordinates (0-1)
        pdf_source: Path to PDF file, or PDF bytes
    Returns:
        images, page_numbers - List of PIL Images with highlighted regions, page number for each image
    """
    if isinstance(pdf_source, bytes):
        fitz_doc = fitz.Document(stream=pdf_source)
    elif isinstance(pdf_source, str):
        fitz_doc = fitz.Document(pdf_source)
    else:
        raise ValueError('pdf_source must be str path or bytes')
    pages_dims = [fitz_doc[i].rect for i in range(len(fitz_doc))]
    fitz_doc.close()

    page_bboxes = {}
    for citation in citation_bboxes:
        page = citation['page']
        bbox_normed = citation['bbox']
        bbox = [
            bbox_normed[0] * pages_dims[page].width,
            bbox_normed[1] * pages_dims[page].height,
            bbox_normed[2] * pages_dims[page].width,
            bbox_normed[3] * pages_dims[page].height,
        ]
        if page not in page_bboxes:
            page_bboxes[page] = []
        page_bboxes[page].append(bbox)
    images = []
    page_numbers = []
    for page, bboxes in page_bboxes.items():
        img = render_pdf_page_with_highlights(
            pdf_source=pdf_source,
            page=page,
            bboxes=bboxes
        )
        images.append(img)
        page_numbers.append(page)
    return images, page_numbers

#globalutils.citation
async def get_unstract_citation_images(
        pdf_source: str | bytes,
        unstract_response_json: dict,
        citation_query: str,
        citation_prompt: str,
        openai_client: AsyncOpenAI,
        return_page_numbers: bool = False
):
    """
    Generate images for each citation bounding box.

    Args:
        pdf_source: Path to PDF file, or PDF bytes
        unstract_response_json: JSON response from Unstract OCR
        citation_query: Citation text to search for
        citation_prompt: LLM prompt for finding citation lines
        openai_client: AsyncOpenAI client instance
        return_page_numbers: Whether to return page numbers along with images
    Returns:
        List of PIL Images with highlighted regions, or (images (list), page_numbers (list)) tuple if return_page_numbers is True
    """
    citation_bboxes = await find_citation_bboxes_normed(
        unstract_response_json=unstract_response_json,
        citation_query=citation_query,
        citation_prompt = citation_prompt,
        openai_client=openai_client
    )
    if citation_bboxes is None:
        return None

    images, page_numbers = render_pdf_bboxes_to_images(
        citation_bboxes=citation_bboxes,
        pdf_source=pdf_source
    )
    if return_page_numbers:
        return images, page_numbers
    return images

# globalutils.st_file_serving
class StreamlitPDFServer:
    # note - for this to work, you must set server.enableStaticServing = true in the streamlit config
    def __init__(self, pdf_path):
        self.source_pdf_path = pdf_path
        self.dest_pdf_name = None
        self.dest_pdf_path = None

    def serve_pdf(self):
        ts = str(time.time()).replace('.', '')
        basename = os.path.basename(self.source_pdf_path)
        safe_basename = re.sub(r'\s+', '_', basename)
        self.dest_pdf_name = ts + safe_basename

        self.dest_pdf_path = os.path.join('static', self.dest_pdf_name)
        shutil.copy(self.source_pdf_path, self.dest_pdf_path)
        return f"app/static/{self.dest_pdf_name}"

    def get_page_link(self, page = None):
        '''Get link to the PDF, optionally at a specific page (0-indexed).'''
        if self.dest_pdf_path is None:
            self.serve_pdf()

        link = f"app/static/{self.dest_pdf_name}"
        if page is not None:
            link += f"#page={page+1}"
        return link

    def destroy(self):
        if self.dest_pdf_path and os.path.exists(self.dest_pdf_path):
            os.remove(self.dest_pdf_path)


# globalutils.openai_uploading
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

async def get_or_upload_async(file_path: str, client: AsyncOpenAI, cache_path: str, purpose = "user_data") -> str:
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