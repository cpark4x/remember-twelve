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

## Quick Start: Curate from Google Photos

The complete Google Photos integration is now live! Curate your best 12 photos from any year directly from Google Photos.

### Setup (One-Time)

1. **Get Google Cloud Credentials:**
   ```bash
   # 1. Go to https://console.cloud.google.com
   # 2. Create project "Remember Twelve"
   # 3. Enable Google Photos Library API
   # 4. Create OAuth 2.0 credentials (Desktop app)
   # 5. Download as google_photos_credentials.json
   ```

2. **Install Dependencies:**
   ```bash
   cd ~/dev/projects/remember-twelve
   pip install -r requirements.txt
   ```

### Curate Photos

```bash
# Curate your best 12 photos from 2023
python curate_from_google_photos.py --year 2023

# Use different strategy
python curate_from_google_photos.py --year 2024 --strategy people_first

# Strategies: balanced, aesthetic_first, people_first, top_heavy
```

The script will:
1. ✅ Authenticate with Google Photos (browser opens)
2. ✅ Fetch all photos from specified year
3. ✅ Analyze quality + emotional significance
4. ✅ Apply AI curation algorithm
5. ✅ Select best 12 photos with diversity
6. ✅ Save results with Google Photos links

Results saved to `twelve_{year}_{strategy}.json` with clickable Google Photos links!

---

## Implementation Status

### Completed Features ✅

**Epic 1: AI-Powered Photo Curation Engine**
- ✅ Feature 1.1: Photo Quality Analyzer (Phase 1 & 2)
- ✅ Feature 1.2: Emotional Significance Detector (Phase 1 & 2)
- ✅ Feature 1.4: Twelve Curation Engine
- ✅ **Feature 1.5: Google Photos Integration (Phase 1-3)** ← NEW!

**What Works Now:**
- End-to-end curation from Google Photos
- OAuth 2.0 authentication with encrypted token storage
- AI quality analysis (sharpness, exposure, composition)
- Emotional significance detection (faces, emotions, intimacy)
- Balanced temporal distribution across months
- Visual diversity filtering
- Results with Google Photos links

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
