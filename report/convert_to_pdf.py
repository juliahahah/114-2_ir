import markdown
import os
import subprocess

report_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(report_dir, 'midterm_report.md'), 'r', encoding='utf-8') as f:
    md_content = f.read()

html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite', 'toc'])

# Resolve image paths to absolute file paths for the browser
figures_abs = os.path.join(report_dir, 'figures').replace('\\', '/')
html_body = html_body.replace('src="figures/', f'src="file:///{figures_abs}/')

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<style>
@page {{
    size: A4;
    margin: 2cm 2.5cm;
}}
body {{
    font-family: "Microsoft JhengHei", "PingFang TC", sans-serif;
    font-size: 11pt;
    line-height: 1.8;
    color: #222;
    max-width: 100%;
}}
h1 {{
    font-size: 20pt;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 8px;
    color: #1a1a2e;
    border-bottom: 3px solid #16213e;
    padding-bottom: 8px;
}}
h2 {{
    font-size: 15pt;
    color: #16213e;
    border-bottom: 2px solid #0f3460;
    padding-bottom: 5px;
    margin-top: 30px;
    page-break-after: avoid;
}}
h3 {{
    font-size: 13pt;
    color: #0f3460;
    margin-top: 20px;
    page-break-after: avoid;
}}
h4 {{
    font-size: 11.5pt;
    color: #333;
    margin-top: 15px;
}}
blockquote {{
    border-left: 4px solid #0f3460;
    padding: 10px 15px;
    margin: 15px 0;
    background: #f0f4f8;
    font-size: 10.5pt;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}}
th {{
    background-color: #16213e;
    color: white;
    padding: 8px 10px;
    text-align: left;
    font-weight: bold;
}}
td {{
    padding: 6px 10px;
    border: 1px solid #ccc;
}}
tr:nth-child(even) {{
    background-color: #f8f9fa;
}}
img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 15px auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    page-break-inside: avoid;
}}
code {{
    background: #f4f4f4;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 9.5pt;
    font-family: Consolas, "Courier New", monospace;
}}
pre {{
    background: #2d2d2d;
    color: #f8f8f2;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.5;
    page-break-inside: avoid;
}}
pre code {{
    background: none;
    padding: 0;
    color: #f8f8f2;
}}
strong {{
    color: #16213e;
}}
hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 20px 0;
}}
p {{
    text-align: justify;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

html_path = os.path.join(report_dir, 'midterm_report.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"HTML generated: {html_path}")

# Use Edge headless to convert to PDF
output_path = os.path.join(report_dir, 'midterm_report.pdf')
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
html_uri = 'file:///' + html_path.replace('\\', '/')

result = subprocess.run([
    edge,
    '--headless',
    '--disable-gpu',
    '--no-sandbox',
    f'--print-to-pdf={output_path}',
    '--print-to-pdf-no-header',
    html_uri
], capture_output=True, text=True, timeout=60)

if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    size_kb = os.path.getsize(output_path) / 1024
    print(f"PDF generated: {output_path} ({size_kb:.1f} KB)")
else:
    print(f"PDF generation failed. stderr: {result.stderr}")
