# Documentation Development

This guide explains how to develop and maintain the Aura documentation website.

## Overview

The Aura documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. The documentation is hosted on GitHub Pages and is automatically deployed when changes are pushed to the main branch.

## Local Development

To develop the documentation locally, follow these steps:

1. Install the required dependencies:
   ```bash
   pip install mkdocs mkdocs-material
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/Sentientia/Aura.git
   cd Aura
   ```

3. Start the local development server:
   ```bash
   mkdocs serve
   ```

4. Open your browser and navigate to `http://localhost:8000` to see the documentation.

5. Make changes to the Markdown files in the `docs/` directory and see the changes reflected in real-time.

## Documentation Structure

The documentation is organized as follows:

- `docs/index.md`: The homepage of the documentation.
- `docs/installation.md`: Installation instructions.
- `docs/architecture.md`: System architecture overview.
- `docs/agents/`: Documentation for the agent components.
- `docs/actions/`: Documentation for the action components.
- `docs/ui.md`: Documentation for the user interface.
- `docs/contributing.md`: Contributing guidelines.
- `docs/docs_development.md`: This guide for documentation development.

## Adding New Pages

To add a new page to the documentation:

1. Create a new Markdown file in the appropriate directory.
2. Add the page to the navigation in `mkdocs.yml`:
   ```yaml
   nav:
     - Home: index.md
     - ...
     - Your New Page: your_new_page.md
   ```

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch. The deployment is handled by a GitHub Actions workflow defined in `.github/workflows/mkdocs.yml`.

To manually deploy the documentation:

1. Build the documentation:
   ```bash
   mkdocs build
   ```

2. Deploy to GitHub Pages:
   ```bash
   mkdocs gh-deploy
   ```

## Best Practices

When writing documentation, follow these best practices:

- Use clear and concise language.
- Include code examples where appropriate.
- Use headings to organize content.
- Include links to related documentation.
- Use images and diagrams to illustrate complex concepts.
- Keep the documentation up-to-date with the codebase.

## Troubleshooting

If you encounter issues with the documentation:

1. Check the MkDocs logs for errors.
2. Verify that the Markdown syntax is correct.
3. Ensure that all links and images are valid.
4. Check that the navigation in `mkdocs.yml` is correctly formatted.