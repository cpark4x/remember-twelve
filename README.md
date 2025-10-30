# Remember Twelve

**Transform the fleeting chaos of digital photos into lasting rituals of reflection and connection.**

**North Star**: *"Preserve your year in twelve unforgettable moments — for every circle that matters."*

---

## Project Overview

Remember Twelve is an AI-powered memory preservation app that automatically curates twelve meaningful photos per year for different "circles" (Family, Individual, Kids, Extended Family, etc.). Unlike manual photo books or generic cloud storage, Remember Twelve combines intelligent curation with multi-archive organization to create lasting, revisitable memory artifacts.

**Repository**: https://github.com/cpark4x/remember-twelve

---

## Spec-Driven Development Structure

This project follows a spec-driven product development approach with four layers:

### 1-vision/ — Product Foundation
- [Vision.md](1-vision/Vision.md) - Product vision and North Star
- [ProblemStatement.md](1-vision/ProblemStatement.md) - Problem analysis
- [Principles.md](1-vision/Principles.md) - 7 core product principles
- [SuccessMetrics.md](1-vision/SuccessMetrics.md) - Success metrics and targets

### 2-product/ — Epics & Features
- [Epic 1: AI-Powered Photo Curation Engine](2-product/Epics/epic-1-ai-curation-engine.md)
- [Epic 2: Multi-Circle Memory Organization](2-product/Epics/epic-2-multi-circle-organization.md)
- [Epic 3: Reflection & Archive Experience](2-product/Epics/epic-3-reflection-archive-experience.md)

### 3-design/ — Design System
- [DesignVision.md](3-design/DesignVision.md) - Design philosophy, visual style, accessibility

### 4-technology/ — Architecture
- [Architecture.md](4-technology/Architecture.md) - System architecture, tech stack, infrastructure

---

## Toolkits

This project uses symlinked toolkits for reusability across projects:

- **amplifier** → `~/dev/toolkits/amplifier` - AI development environment with specialized agents
- **templates** → `~/dev/toolkits/templates` - Reusable document templates

All vision documents follow templates from `~/dev/toolkits/templates`:
- Vision.md follows [vision-template.md](templates/vision-template.md)
- ProblemStatement.md follows [problem-statement-template.md](templates/problem-statement-template.md)
- Principles.md follows [principles-template.md](templates/principles-template.md)
- Epics follow [epic-template.md](templates/epic-template.md)

---

## Key Product Principles

1. **Effortless by Default, Control When Needed** - Zero configuration required, but full override available
2. **Preservation Over Perfection** - Automatic curation beats manual perfection that never happens
3. **Multi-Circle by Design** - Every feature works across multiple memory archives
4. **Timeless Design, Future-Proof Data** - Built to last decades, not just years
5. **AI as Curator, Not Creator** - AI selects, never fabricates or manipulates
6. **Reflection is the Product** - Optimize for revisiting, not just storage
7. **Start Simple, Scale Complexity** - Minimal V1, progressive disclosure

---

## Tech Stack

| Layer        | Technology                          |
| ------------ | ----------------------------------- |
| **Frontend** | Swift (iOS), Kotlin (Android), React (Web) |
| **Backend**  | Python (FastAPI), Node.js           |
| **Database** | PostgreSQL, Redis                   |
| **Storage**  | AWS S3, CloudFront CDN              |
| **ML**       | Python, TensorFlow, OpenCV          |
| **Infra**    | Docker, Kubernetes, GitHub Actions  |

---

## Using Amplifier with This Project

To leverage amplifier's specialized agents while working on Remember Twelve:

```bash
claude --add-dir ~/dev/toolkits/amplifier --add-dir ~/dev/projects/remember-twelve
```

Then use agents like:
- `zen-architect` for design decisions
- `modular-builder` for implementation
- `concept-extractor` for analyzing specs

---

## Quick Start: Curate Your Photos

Remember Twelve works with any photo collection - local files, Google Takeout exports, Apple Photos, or camera rolls.

### Option A: Apple Photos (Easiest for Apple users) 🍎

**Automated export on macOS:**

```bash
# Automatically export and curate 2023 photos
./sync_photos.sh --apple-photos 2023

# Export specific album
./sync_photos.sh --apple-photos 2023 --album "Favorites"

# Show manual instructions
./sync_photos.sh --apple-photos 2023 --manual
```

**Or manual export (all platforms):**
1. Open Photos app
2. Filter by year: Search → `date:2023`
3. Select all (`Cmd + A`)
4. File → Export → Export Unmodified Originals
5. Run: `./sync_photos.sh ~/Desktop/Photos2023 2023`

📖 **Complete guide**: See [docs/APPLE_PHOTOS_GUIDE.md](docs/APPLE_PHOTOS_GUIDE.md)

### Option B: Google Takeout (For Google Photos users)

**⚠️ NOTE**: Google deprecated their Photos API in March 2025. Direct API access no longer works. Use Google Takeout instead.

1. **Export Your Google Photos**:
   - Go to https://takeout.google.com
   - Select only "Google Photos"
   - Filter by year (e.g., 2023)
   - Download and extract the ZIP

2. **Curate Your Photos**:
   ```bash
   # Point to your extracted Takeout folder
   ./sync_photos.sh ~/Downloads/Takeout/Google\ Photos 2023

   # Or use Python directly
   python curate_from_google_photos.py \
       --source ~/Downloads/Takeout/Google\ Photos \
       --year 2023 \
       --strategy balanced \
       --flexible-months
   ```

📖 **Complete guide**: See [docs/GOOGLE_TAKEOUT_GUIDE.md](docs/GOOGLE_TAKEOUT_GUIDE.md)

### Option C: Local Photo Folder

Works with any organized photo collection:

```bash
# Curate from any local folder
./sync_photos.sh ~/Pictures/2023 2023

# Or be explicit
python curate_from_google_photos.py \
    --source ~/Pictures/2023 \
    --year 2023 \
    --strategy balanced
```

### Option D: Google Drive Sync

If you have Google Drive desktop app syncing your Photos:

```bash
./sync_photos.sh ~/GoogleDrive/Google\ Photos 2023
```

### Curation Strategies

Available strategies for different preferences:

- `balanced` - Even quality + emotional significance (default)
- `aesthetic_first` - Prioritize visual quality
- `people_first` - Prioritize faces and emotions
- `top_heavy` - More photos from best months

Add `--flexible-months` to fill empty months with next-best photos.

### The Curation Process

1. ✅ Scan photo folder for specified year
2. ✅ Analyze quality (sharpness, exposure, composition)
3. ✅ Detect emotional significance (faces, emotions, intimacy)
4. ✅ Apply AI curation algorithm with selected strategy
5. ✅ Select best 12 photos with diversity
6. ✅ Distribute across 12 months (flexibly if requested)
7. ✅ Save results to `ui/photos_data.json`

### View Your Remember Twelve

```bash
cd ui
python3 sync_server.py

# Open browser: http://localhost:8765
```

### Setup (One-Time)

```bash
cd ~/dev/projects/remember-twelve
pip install -r requirements.txt
```

### Why Not Direct Google Photos API?

See [GOOGLE_PHOTOS_DEPRECATION.md](GOOGLE_PHOTOS_DEPRECATION.md) for details on why the API no longer works and why local sources are better.

---

## Implementation Status

### Completed Features ✅

**Epic 1: AI-Powered Photo Curation Engine**
- ✅ Feature 1.1: Photo Quality Analyzer (Phase 1 & 2)
- ✅ Feature 1.2: Emotional Significance Detector (Phase 1 & 2)
- ✅ Feature 1.4: Twelve Curation Engine
- ✅ Feature 1.5: Local File Source Integration (Phase 1-3)
- ⚠️ ~~Google Photos API Integration~~ (deprecated by Google March 2025)

**What Works Now:**
- End-to-end curation from any local photo source
- Support for Google Takeout exports (includes rich metadata)
- Support for any organized photo folder
- AI quality analysis (sharpness, exposure, composition)
- Emotional significance detection (faces, emotions, intimacy)
- Balanced temporal distribution across months
- Visual diversity filtering
- **Flexible month distribution** (fills empty months with best available photos)
- **Web viewer** for browsing curated photos
- **Shell script bridge** for easy automation
- Multiple curation strategies (balanced, aesthetic, people-first, etc.)

### Next Steps

**Epic 2: Multi-Circle Memory Organization**
- Feature 2.1: Circle Creation & Management
- Feature 2.2: Photo-to-Circle Assignment
- Feature 2.3: Per-Circle Curation

**Epic 3: Reflection & Archive Experience**
- Feature 3.1: Year-in-Review Generator
- Feature 3.2: Memory Timeline
- Feature 3.3: Export & Sharing

---

## License

[To be determined]

---

**Built with spec-driven development, amplifier patterns, and reusable templates.**
