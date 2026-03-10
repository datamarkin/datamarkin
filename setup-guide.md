# Datamarkin

## Requirements

- macOS (Apple Silicon)
- Python 3.13+
- Conda (recommended) or pip

## Setup

```bash
# Create environment
conda create -n datamarkin python=3.13 -y
conda activate datamarkin

# Install dependencies
pip install -r requirements.txt

# For building .app bundle
pip install py2app
```

## Run (Development)

### Option 1: Flask Server (Browser Access)
```bash
python run_server.py
```
Starts a Flask server on `127.0.0.1:5001`. Open your browser to `http://localhost:5001` to access the app.

### Option 2: PyWebView (Native Window)
```bash
python main.py
```
Starts a Flask server on `127.0.0.1:5001` and opens a native PyWebView window. The database and data directory (`~/Datamarkin/`) are created automatically on first run.

## Seed Sample Data

```bash
# Seed database with sample projects and download images from picsum.photos
python scripts/seed.py

# Fresh seed (deletes existing DB and project folders first)
python scripts/seed.py --fresh
```

## Build macOS .app

```bash
python build_macos.py py2app
```

The `.app` bundle is generated at:

```
dist/Datamarkin.app
```

To run it directly:

```bash
open dist/Datamarkin.app
```

### Create DMG

After building the `.app`, create a distributable `.dmg`:

```bash
hdiutil create -volname "Datamarkin" \
  -srcfolder dist/Datamarkin.app \
  -ov -format UDZO \
  dist/Datamarkin-<version>.dmg
```

The `.dmg` file is at `dist/Datamarkin-<version>.dmg`.

## Versioning

Use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR** — breaking changes or major rewrites
- **MINOR** — new features (e.g., training, model zoo, workflows)
- **PATCH** — bug fixes and small improvements

Current version: `0.1.0` (scaffold)

Update the version in `config.py` when releasing:

```python
APP_VERSION = "0.1.0"
```

Tag releases in git:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Project Structure

```
datamarkin/
  main.py                 ← entry point (Flask + PyWebView)
  app.py                  ← Flask app factory
  config.py               ← paths, constants
  db.py                   ← SQLite schema + helpers
  scripts/
    seed.py               ← seed DB with sample data + images
  build_macos.py          ← py2app build script
  requirements.txt
  routes/
    studio_routes.py      ← page routes (HTML)
    api_routes.py         ← JSON API routes (future)
  templates/
    base.html           ← sidebar layout
    projects.html       ← project list
  static/
    css/
      bulma.min.css       ← Bulma CSS framework
      app.css             ← custom styles
    js/                   ← JS files (future)
```

## Data Storage

All user data is stored locally:

```
~/Datamarkin/
  datamarkin.db           ← SQLite database
  projects/
    <project-uuid>/
      images/
        <image-uuid>.jpg
```

## Tech Stack

| Component     | Technology         |
|---------------|--------------------|
| Desktop shell | PyWebView          |
| Backend       | Flask              |
| Templates     | Jinja2             |
| Frontend      | Vanilla JS, Bulma  |
| Database      | SQLite             |
| Annotation    | MarkinJS, SAM2     |
| Training      | PyTorch (MPS)      |

