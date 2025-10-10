# Templates Guide — Remember Twelve

This document explains how Remember Twelve uses the reorganized templates toolkit.

---

## Template Organization

As of October 2025, templates have been reorganized into folders that mirror the spec-driven product development structure.

### Before (Flat Structure)
```
templates/
├── vision-template.md
├── epic-template.md
├── metric-template.md
└── ... (all files in one folder)
```

### After (Organized Structure)
```
templates/
├── 1-vision/
│   ├── vision-template.md
│   ├── problem-statement-template.md
│   ├── principles-template.md
│   └── success-metrics-template.md
├── 2-product/
│   ├── epic-template.md
│   ├── feature-template.md
│   └── userstory-template.md
├── 3-design/
│   ├── design-vision-template.md
│   ├── interaction-flow-template.md
│   ├── prototype-links-template.md
│   ├── component-template.md
│   ├── pattern-template.md
│   └── workflow-template.md
├── 4-technology/
│   ├── architecture-template.md
│   └── stack-decision-log-template.md
└── other/
    ├── readme-template.md
    ├── sitemap-template.md
    └── routes-template.json
```

---

## How Remember Twelve Uses Templates

### Documents Created from Templates

| Remember Twelve Document | Source Template |
|-------------------------|-----------------|
| [1-vision/Vision.md](1-vision/Vision.md) | [templates/1-vision/vision-template.md](templates/1-vision/vision-template.md) |
| [1-vision/ProblemStatement.md](1-vision/ProblemStatement.md) | [templates/1-vision/problem-statement-template.md](templates/1-vision/problem-statement-template.md) |
| [1-vision/Principles.md](1-vision/Principles.md) | [templates/1-vision/principles-template.md](templates/1-vision/principles-template.md) |
| [1-vision/SuccessMetrics.md](1-vision/SuccessMetrics.md) | [templates/1-vision/success-metrics-template.md](templates/1-vision/success-metrics-template.md) |
| [2-product/Epics/*.md](2-product/Epics/) | [templates/2-product/epic-template.md](templates/2-product/epic-template.md) |
| [3-design/DesignVision.md](3-design/DesignVision.md) | [templates/3-design/design-vision-template.md](templates/3-design/design-vision-template.md) |
| [3-design/InteractionFlow.md](3-design/InteractionFlow.md) | [templates/3-design/interaction-flow-template.md](templates/3-design/interaction-flow-template.md) |
| [3-design/PrototypeLinks.md](3-design/PrototypeLinks.md) | [templates/3-design/prototype-links-template.md](templates/3-design/prototype-links-template.md) |
| [4-technology/Architecture.md](4-technology/Architecture.md) | [templates/4-technology/architecture-template.md](templates/4-technology/architecture-template.md) |

---

## Benefits of Organized Structure

1. **Clear Discovery**: "I need a vision doc? Look in templates/1-vision/"
2. **Matches Projects**: Template folders mirror project structure
3. **Scalable**: Easy to add more templates without clutter
4. **Intent-Driven**: Organization shows product development workflow

---

## Using Templates for New Documents

### Creating a New Epic

```bash
# Copy the template
cp templates/2-product/epic-template.md 2-product/Epics/epic-4-new-feature.md

# Edit with product-specific content
# Follow the template structure
```

### Creating a New Feature

```bash
cp templates/2-product/feature-template.md 2-product/Features/feature-1.1-photo-quality.md
```

---

## Template Maintenance

Templates live in `~/dev/toolkits/templates` and are symlinked into projects.

**When you improve a template**:
1. Edit the master template in `~/dev/toolkits/templates/[layer]/[template].md`
2. Changes automatically available to all future projects
3. Existing projects see updated templates through symlink

**Example**:
```bash
# Edit master template
vim ~/dev/toolkits/templates/1-vision/vision-template.md

# All projects with symlinked templates now see the update
ls -la ~/dev/projects/*/templates/1-vision/vision-template.md
```

---

## Why This Matters

Remember Twelve pioneered this organized template structure. As you create more products:

- **Consistent Quality**: Every product starts with proven templates
- **Faster Iteration**: Copy template → fill in → ship
- **Knowledge Preservation**: Best practices baked into templates
- **Continuous Improvement**: Refine templates as you learn

---

**This approach makes spec-driven product development scalable across an entire portfolio of products.**
