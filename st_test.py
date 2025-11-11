import streamlit as st
import fitz
from PIL import Image
import io


def render_pdf_page_with_highlight(
        pdf_path: str,
        page: int,
        bbox: list[float],
        highlight_color: tuple[float, float, float] = (1, 1, 0),  # RGB 0-1, default yellow
        highlight_opacity: float = 0.3,
        zoom: float = 2.0  # Higher = better quality, 2.0 is good default
) -> Image.Image:
    """
    Render a PDF page with a highlighted bounding box.

    Args:
        pdf_path: Path to PDF file
        page: Page number (0-indexed)
        bbox: Bounding box as [x0, y0, x1, y1] in PDF coordinates
        highlight_color: RGB tuple with values 0-1
        highlight_opacity: Transparency of highlight (0-1)
        zoom: Rendering resolution multiplier

    Returns:
        PIL Image with highlighted region
    """
    doc = fitz.open(pdf_path)

    if page < 0 or page >= len(doc):
        raise ValueError(f"Page {page} out of range. PDF has {len(doc)} pages.")

    pdf_page = doc[page]

    # Render page at higher resolution
    mat = fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat)

    # Convert to PIL Image
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    # Draw highlight rectangle
    # Note: bbox coordinates need to be scaled by zoom factor
    x0, y0, x1, y1 = bbox
    scaled_bbox = [x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom]

    # Create a semi-transparent overlay
    from PIL import ImageDraw
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Convert RGB 0-1 to 0-255
    color_255 = tuple(int(c * 255) for c in highlight_color)
    alpha = int(highlight_opacity * 255)
    fill_color = (*color_255, alpha)

    # Draw filled rectangle
    draw.rectangle(scaled_bbox, fill=fill_color, outline=(255, 0, 0, 255), width=3)

    # Composite the overlay onto the original image
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)

    doc.close()

    return img.convert('RGB')  # Convert back to RGB for st.image

page_input = st.number_input('Page number (0-indexed)', min_value=0, value=0)
bbox_input = st.text_input('Bounding box [x0, y0, x1, y1]', '100, 100, 300, 300')
bbox = [float(coord) for coord in bbox_input.split(',')]
pdf_path = r"C:\Users\jaeckle\PycharmProjects\DavisBaconApp\documents\coulson.pdf"

if st.button('render highlight'):
    highlighted_img = render_pdf_page_with_highlight(
        pdf_path=pdf_path,
        page=page_input,
        bbox=bbox,
        highlight_color=(1, 1, 0),  # Yellow
        highlight_opacity=0.3,
        zoom=2.0
    )
    st.image(highlighted_img, caption=f'Page {page_input} with Highlight')