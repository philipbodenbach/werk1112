"""Bounded document extraction shared by every Werk text/vision backend.

No model imports, network access, macros, external XML entities or archive writes.
Input arrives on stdin; all intermediate PDF/office data stays in this process.
"""
import base64
import io
import json
import re
import sys
import os
import subprocess
import threading
import time
import zipfile
from xml.etree import ElementTree as ET

MAX_TEXT = 8 * 1024 * 1024
MAX_OUTPUT = 60 * 1024 * 1024
MAX_PAGES = 200


def own_children():
    """OCR descendants must terminate with their worker, including on Windows."""
    if os.name != "nt":
        parent = os.getppid()
        def watch():
            while os.getppid() == parent:
                time.sleep(0.5)
            if os.getpgrp() == os.getpid():
                os.killpg(os.getpid(), 9)
            os._exit(1)
        threading.Thread(target=watch, daemon=True).start()
        return None
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
            ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
            ("processes", wintypes.DWORD), ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    limits = Extended()
    limits.basic.flags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
    limits.job_memory = 2 * 1024**3
    if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise ValueError("document worker cannot establish Windows child-process ownership")
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    parent = kernel.OpenProcess(0x00100000, False, os.getppid())  # SYNCHRONIZE
    if not parent:
        raise ValueError("document worker cannot watch its parent")
    def watch_parent():
        kernel.WaitForSingleObject(parent, 0xFFFFFFFF)
        os._exit(1)
    threading.Thread(target=watch_parent, daemon=True).start()
    # Process exit closes the last non-inheritable handle and kills descendants.
    return job


def xml(data):
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("document XML entities are not supported")
    return ET.fromstring(data)


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def text_nodes(root):
    output = []
    for element in root.iter():
        if tag(element) in ("t", "p", "h", "span") and element.text:
            output.append(element.text)
        if tag(element) in ("p", "br", "tab"):
            output.append("\n" if tag(element) != "tab" else "\t")
        if element.tail:
            output.append(element.tail)
    return "".join(output)


def office(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 10000 or sum(e.file_size for e in entries) > 64 * 1024 * 1024:
            raise ValueError("office archive exceeds extraction limits")
        names = {e.filename for e in entries}
        if len(names) != len(entries) or any(e.flag_bits & 1 for e in entries):
            raise ValueError("encrypted or duplicate office entries are not supported")

        def read(name):
            info = archive.getinfo(name)
            if info.file_size > 16 * 1024 * 1024:
                raise ValueError("office part exceeds 16 MiB")
            return xml(archive.read(name))

        if "word/document.xml" in names:
            text = text_nodes(read("word/document.xml"))
            for name in sorted(names):
                if re.fullmatch(r"word/(?:footnotes|endnotes|header\d+|footer\d+)\.xml", name):
                    text += "\n" + text_nodes(read(name))
            return [dict(text=text, image=None)]
        if "content.xml" in names and "mimetype" in names:
            if not archive.read("mimetype").startswith(b"application/vnd.oasis.opendocument."):
                raise ValueError("unsupported office format")
            return [dict(text=text_nodes(read("content.xml")), image=None)]
        if "ppt/presentation.xml" in names:
            rels = {e.attrib["Id"]: e.attrib["Target"] for e in read("ppt/_rels/presentation.xml.rels")}
            pages = []
            for slide in read("ppt/presentation.xml").iter():
                if tag(slide) != "sldId":
                    continue
                rid = slide.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                target = rels.get(rid, "")
                if not re.fullmatch(r"slides/slide\d+\.xml", target):
                    raise ValueError("unsupported presentation relationship")
                pages.append(dict(text=text_nodes(read("ppt/" + target)), image=None))
            return pages
        if "xl/workbook.xml" in names:
            shared = []
            if "xl/sharedStrings.xml" in names:
                shared = ["".join(e.itertext()) for e in read("xl/sharedStrings.xml")]
            rels = {e.attrib["Id"]: e.attrib["Target"] for e in read("xl/_rels/workbook.xml.rels")}
            pages = []
            for sheet in read("xl/workbook.xml").iter():
                if tag(sheet) != "sheet":
                    continue
                rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                target = rels.get(rid, "").removeprefix("/xl/")
                if not re.fullmatch(r"worksheets/sheet\d+\.xml", target):
                    raise ValueError("unsupported spreadsheet relationship")
                rows = ["Sheet: " + sheet.attrib.get("name", "")]
                for row in read("xl/" + target).iter():
                    if tag(row) != "row":
                        continue
                    cells = []
                    for cell in row:
                        value = "".join(cell.itertext()) if cell.attrib.get("t") == "inlineStr" else next((e.text or "" for e in cell if tag(e) == "v"), "")
                        if cell.attrib.get("t") == "s" and value:
                            value = shared[int(value)]
                        formula = next((e.text for e in cell if tag(e) == "f"), None)
                        if formula:
                            value += " [formula: " + formula + "; cached result]"
                        cells.append(cell.attrib.get("r", "") + "=" + value)
                    rows.append("\t".join(cells))
                pages.append(dict(text="\n".join(rows), image=None))
            return pages
        raise ValueError("unsupported archive; expected DOCX, PPTX, XLSX or OpenDocument")


def extract(data, options):
    if data.startswith(b"%PDF-"):
        try:
            from pypdf import PdfReader
            import pypdf.filters as filters
            for name in ("ZLIB_MAX_OUTPUT_LENGTH", "LZW_MAX_OUTPUT_LENGTH", "RUN_LENGTH_MAX_OUTPUT_LENGTH", "JBIG2_MAX_OUTPUT_LENGTH"):
                if hasattr(filters, name):
                    setattr(filters, name, min(getattr(filters, name), 8 * 1024 * 1024))
        except ImportError as e:
            raise ValueError("PDF text extraction requires pypdf in WERK_DOCUMENT_PYTHON") from e
        pdf = PdfReader(io.BytesIO(data))
        if pdf.is_encrypted:
            raise ValueError("encrypted PDFs are not supported")
        if not 1 <= len(pdf.pages) <= MAX_PAGES:
            raise ValueError("PDF exceeds 200 pages")
        renderer = None
        if options.get("vision") or options.get("ocr"):
            try:
                import pypdfium2
            except ImportError as e:
                raise ValueError("visual PDFs and OCR require pypdfium2 in WERK_DOCUMENT_PYTHON") from e
            renderer = pypdfium2.PdfDocument(data)
        pages = []
        total = 0
        try:
            for index, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                image = None
                if renderer is not None:
                    rendered_page = renderer[index]
                    try:
                        width, height = rendered_page.get_size()
                        if min(width, height) <= 0:
                            raise ValueError("invalid PDF page size")
                        bitmap = rendered_page.render(scale=min(2.0, 1600 / max(width, height)))
                        try:
                            out = io.BytesIO()
                            bitmap.to_pil().save(out, format="PNG")
                            if options.get("vision"):
                                image = base64.b64encode(out.getvalue()).decode("ascii")
                            if options.get("ocr"):
                                try:
                                    result = subprocess.run([os.environ.get("WERK_DOCUMENT_TESSERACT", "tesseract"),
                                        "stdin", "stdout", "-l", os.environ.get("WERK_DOCUMENT_OCR_LANG", "eng")],
                                        input=out.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=True)
                                except FileNotFoundError as error:
                                    raise ValueError("OCR requires Tesseract; configure WERK_DOCUMENT_TESSERACT") from error
                                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                                    raise ValueError("OCR failed or exceeded its page time limit") from error
                                recognized = result.stdout.decode("utf-8", errors="strict").strip()
                                if recognized:
                                    text += "\n[OCR text]\n" + recognized
                        finally:
                            bitmap.close()
                    finally:
                        rendered_page.close()
                if not options.get("vision") and not text.strip():
                    raise ValueError("PDF contains a page without extractable text; use a vision model or OCR the document first")
                total += len(text.encode()) + (len(image) if image else 0)
                if total > MAX_OUTPUT:
                    raise ValueError("expanded PDF exceeds output limit")
                pages.append(dict(text=text, image=image))
        finally:
            if renderer is not None:
                renderer.close()
        representation = "text and page images" if options.get("vision") else "extracted text; visual page content omitted"
        if options.get("ocr"):
            representation += "; OCR enabled"
    elif data.startswith(b"PK\x03\x04"):
        pages = office(data)
        representation = "extracted text and table cells; embedded images and layout omitted"
    else:
        raise ValueError("unsupported binary document")
    if not 1 <= len(pages) <= MAX_PAGES:
        raise ValueError("document exceeds 200 pages/sheets/slides")
    if sum(len(p["text"].encode()) for p in pages) > MAX_TEXT:
        raise ValueError("extracted document text exceeds 8 MiB")
    return dict(pages=pages, representation=representation)


def main():
    ownership = own_children()
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (75, 75))
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    data = sys.stdin.buffer.read(50 * 1024 * 1024 + 1)
    if len(data) > 50 * 1024 * 1024:
        raise ValueError("document exceeds 50 MiB")
    output = json.dumps(extract(data, json.loads(sys.argv[1])), ensure_ascii=False)
    if len(output.encode()) > MAX_OUTPUT:
        raise ValueError("document output exceeds 60 MiB")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # No traceback, local paths or document body in the public error.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(message[:1000], file=sys.stderr)
        sys.exit(1)
