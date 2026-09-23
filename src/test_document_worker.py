import importlib.util
import io
from pathlib import Path
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("document_worker", Path(__file__).with_name("document_worker.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def archive(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as z:
        for name, body in files.items():
            z.writestr(name, body)
    return output.getvalue()


def pdf(text="Werk document 42"):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject({NameObject("/Type"):NameObject("/Font"),NameObject("/Subtype"):NameObject("/Type1"),NameObject("/BaseFont"):NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):DictionaryObject({NameObject("/F1"):writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class Documents(unittest.TestCase):
    def test_pdf_text_is_independent_of_model_and_accelerator(self):
        result = worker.extract(pdf(), {"vision":False})
        self.assertIn("Werk document 42", result["pages"][0]["text"])
        self.assertIsNone(result["pages"][0]["image"])
        self.assertIn("omitted", result["representation"])

    def test_pdf_visual_preserves_text_and_renders_page(self):
        result = worker.extract(pdf(), {"vision":True})
        import base64
        self.assertTrue(base64.b64decode(result["pages"][0]["image"]).startswith(b"\x89PNG"))
        self.assertIn("Werk document 42", result["pages"][0]["text"])

    def test_empty_pdf_text_is_not_silently_lost(self):
        with self.assertRaisesRegex(ValueError, "without extractable text"):
            worker.extract(pdf(""), {})

    def test_ocr_is_explicit_and_works_without_a_vision_model(self):
        from unittest.mock import patch
        import subprocess
        with patch.object(worker.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"Scanned text 42", b"")) as run:
            result = worker.extract(pdf(""), {"ocr":True, "vision":False})
        self.assertIn("[OCR text]\nScanned text 42", result["pages"][0]["text"])
        self.assertIsNone(result["pages"][0]["image"])
        self.assertTrue(run.call_args.kwargs["input"].startswith(b"\x89PNG"))
        self.assertEqual(run.call_args.kwargs["timeout"], 15)

    def test_ocr_missing_dependency_is_an_explicit_error(self):
        from unittest.mock import patch
        with patch.object(worker.subprocess, "run", side_effect=FileNotFoundError()):
            with self.assertRaisesRegex(ValueError, "OCR requires Tesseract"):
                worker.extract(pdf(""), {"ocr":True})

    def test_docx_preserves_body_and_footnotes(self):
        result = worker.extract(archive({"word/document.xml":"<doc><p><r><t>Hello</t></r></p></doc>",
            "word/footnotes.xml":"<doc><p><r><t>Footnote</t></r></p></doc>"}), {})
        self.assertIn("Hello", result["pages"][0]["text"])
        self.assertIn("Footnote", result["pages"][0]["text"])

    def test_presentation_uses_declared_order_not_filename_order(self):
        data = archive({"ppt/presentation.xml":'<p xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sldId r:id="r2"/><sldId r:id="r1"/></p>',
            "ppt/_rels/presentation.xml.rels":'<rels><rel Id="r1" Target="slides/slide1.xml"/><rel Id="r2" Target="slides/slide2.xml"/></rels>',
            "ppt/slides/slide1.xml":"<slide><t>One</t></slide>","ppt/slides/slide2.xml":"<slide><t>Two</t></slide>"})
        self.assertEqual([p["text"] for p in worker.extract(data,{})["pages"]],["Two","One"])

    def test_spreadsheet_retains_cell_references_and_cached_formula(self):
        data=archive({"xl/workbook.xml":'<book xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheet name="Budget" r:id="r1"/></book>',
            "xl/_rels/workbook.xml.rels":'<rels><rel Id="r1" Target="worksheets/sheet1.xml"/></rels>',
            "xl/sharedStrings.xml":"<strings><si><t>Revenue</t></si></strings>",
            "xl/worksheets/sheet1.xml":'<sheet><row><c r="A1" t="s"><v>0</v></c><c r="B1"><f>2+3</f><v>5</v></c></row></sheet>'})
        text=worker.extract(data,{})["pages"][0]["text"]
        self.assertIn("A1=Revenue",text)
        self.assertIn("B1=5 [formula: 2+3; cached result]",text)

    def test_opendocument(self):
        data=archive({"mimetype":"application/vnd.oasis.opendocument.text","content.xml":"<doc><p>Hello <span>World</span>!</p></doc>"})
        self.assertIn("World!",worker.extract(data,{})["pages"][0]["text"])

    def test_external_xml_entity_is_rejected(self):
        data=archive({"word/document.xml":'<!DOCTYPE doc [<!ENTITY x SYSTEM "file:///etc/passwd">]><doc>&x;</doc>'})
        with self.assertRaisesRegex(ValueError,"entities"):
            worker.extract(data,{})

    def test_arbitrary_zip_is_not_treated_as_a_document(self):
        with self.assertRaisesRegex(ValueError,"unsupported archive"):
            worker.extract(archive({"../outside":"not extracted"}),{})


if __name__ == "__main__":
    unittest.main()
