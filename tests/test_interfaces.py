from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

GENERATE = True


def get_pdf_path():

    pdf_path = Path("./tests/data/2305.03393v1-pg9.pdf")
    return pdf_path


@pytest.fixture
def converter():

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseDocumentBackend
            )
        }
    )

    return converter


def test_convert_path(converter: DocumentConverter):

    pdf_path = get_pdf_path()
    print(f"converting {pdf_path}")

    doc_result = converter.convert(pdf_path)


def test_convert_stream(converter: DocumentConverter):

    pdf_path = get_pdf_path()
    print(f"converting {pdf_path}")

    buf = BytesIO(pdf_path.open("rb").read())
    stream = DocumentStream(name=pdf_path.name, stream=buf)

    doc_result = converter.convert(stream)
    