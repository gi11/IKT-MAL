## Prerequisites

**Pandoc:**

*Linux:* `apt install pandoc`

*Windows:* installer from pandoc website <https://pandoc.org/installing.html>

**Latex:**

*Linux:* `apt install texlive`

*Windows:* download MiKTeX from website <https://miktex.org/download>

**VS Code Extensions**

Install the following vscode extensions

- vscode-pandoc
- Pandoc Markdown Preview

Use `Ctrl+Shift+P` -> `"Open Pandoc Markdown Preview"` when editing .md files for real-time preview.

Use `Ctrl+Shift+P` -> `"Pandoc Render"` -> `"pdf"` to export latex'y pdf

### Install & setup include functionality

Make sure python has been added to PATH. Install the `pandoc-include` filter through pip: `pip install pandoc-include`. 

This allows including .md files in other .md files with the following syntax `!include example_section.md`.

Set the two vscode extensions to use the filter by setting following settings in vscode:

    "pandocMarkdownPreview.extraPandocArguments": "--filter pandoc-include ",
    "pandoc.pdfOptString": "--filter pandoc-include",

## Notes

- The vscode-pandoc extension will not render the pdf if the previous render is open somewhere
- Remember to add a newline at the end of each included file (and between sections in general), as pandoc/markdown can produce some unexpected behaviour if things are not seperated properly with newlines
