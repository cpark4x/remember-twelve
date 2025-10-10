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

## Next Steps

1. **Expand Features**: Create detailed feature specs for each epic
2. **Create User Stories**: Break features into user stories
3. **Design Prototypes**: Build Figma mockups following design vision
4. **Technical Spikes**: Validate ML curation approach, photo library integration
5. **MVP Development**: Start with Epic 1 (AI Curation Engine)

---

## License

[To be determined]

---

**Built with spec-driven development, amplifier patterns, and reusable templates.**
