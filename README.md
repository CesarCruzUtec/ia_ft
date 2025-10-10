# IA FT Project

This project consists of a Python backend (managed with uv) and a Vite frontend, with SAM2 integration.

## Project Structure

```
ia_ft/
├── .vscode/          # VSCode configuration files
├── back/             # Python backend (uv managed)
├── front/            # Vite frontend
├── images/           # Image assets (directory structure tracked, content ignored)
├── models/           # Model files (directory structure tracked, content ignored)
└── meta-sam2/        # SAM2 submodule (Facebook Research)
```

## Prerequisites

- Python 3.x
- [uv](https://github.com/astral-sh/uv) - Python package manager
- Node.js (v16 or higher)
- npm or pnpm
- Git

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd ia_ft
```

### 2. Initialize submodules

```bash
git submodule update --init --recursive
```

### 3. Backend Setup

```bash
cd back
uv sync
```

This will:
- Create a virtual environment
- Install all dependencies from `pyproject.toml`

### 4. Frontend Setup

```bash
cd front
npm install
```

### 5. Running the project

You need to run both services in **separate terminals**.

**Terminal 1 - Backend (FastAPI):**
```bash
cd back
uv run fastapi dev main.py
```
This will start the FastAPI server on `http://localhost:8000`
- API documentation available at: `http://localhost:8000/docs`
- Alternative docs at: `http://localhost:8000/redoc`

**Terminal 2 - Frontend (Vite):**
```bash
cd front
npm run dev
```
This will start the Vite development server (typically on `http://localhost:5173`)

## Development

### Backend

The backend is managed with `uv`. To add dependencies:
```bash
cd back
uv add <package-name>
```

### Frontend

The frontend uses Vite. To add dependencies:
```bash
cd front
npm install <package-name>
```

## Notes

- The `.vscode` folder contains project-specific configurations and should be committed
- The `meta-sam2` directory is a git submodule pointing to Facebook Research's SAM2 repository
- Make sure to run `git submodule update --init --recursive` after cloning to fetch the SAM2 code

## Updating SAM2 Submodule

To update the SAM2 submodule to the latest version:
```bash
cd meta-sam2
git pull origin main
cd ..
git add meta-sam2
git commit -m "Update SAM2 submodule"
```
