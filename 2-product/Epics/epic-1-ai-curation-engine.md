# Epic 1: AI-Powered Photo Curation Engine

---

### Problem Statement

Families take thousands of photos annually but lack the time or framework to identify which moments truly matter. Manual curation requires hours of decision-making that most families never undertake, resulting in digital graveyards of unsorted photos. Users need an intelligent system that automatically identifies the twelve most meaningful photos from each year without requiring manual sorting or configuration.

---

### Objective

Build an AI-powered curation engine that automatically selects twelve meaningful photos per year for each circle, using quality, emotion, diversity, and significance signals—with zero required user configuration but full manual override capability.

---

### Business Value

- **Core Differentiator**: AI curation is the foundational capability that separates Remember Twelve from manual photo book services
- **Effortless Experience**: Delivers on "Preservation Over Perfection" principle—automatic curation removes friction
- **Retention Driver**: Quality curation builds trust; users return when AI selections feel meaningful
- **Scalability**: Automated curation scales to millions of users without linear cost increases

---

### Scope

**In-Scope:**

- Photo quality assessment (sharpness, exposure, composition)
- Emotional significance detection (facial expressions, scene context)
- Temporal diversity (spread across months, avoiding clustering)
- Person/subject diversity (multiple family members, varied activities)
- Manual override system (swap photos, pin favorites)
- Curation confidence scoring and transparency

**Out-of-Scope:**

- Photo editing or manipulation (we curate, not create)
- Real-time curation (annual/monthly batch processing acceptable for V1)
- Multi-modal inputs (video, audio, text) — photos only initially
- Advanced personalization (learning user preferences over time) — V2 feature

---

### Features

This epic comprises the following features:

1. **Feature 1.1: Photo Quality Analyzer** - Assesses technical photo quality (sharpness, exposure, composition)
2. **Feature 1.2: Emotional Significance Detector** - Identifies photos with strong emotional content (smiles, hugs, celebrations)
3. **Feature 1.3: Temporal & Subject Diversity Engine** - Ensures spread across time and subjects
4. **Feature 1.4: Twelve Selection Algorithm** - Combines signals to select optimal twelve photos
5. **Feature 1.5: Manual Override Interface** - Allows users to swap, pin, or exclude photos
6. **Feature 1.6: Curation Transparency Dashboard** - Shows why photos were selected

---

### Success Metrics

**Primary Metrics:**

- **Curation Override Rate**: 10-20% (Goldilocks zone—low enough to show trust, high enough to show engagement)
- **Twelve Completion Rate**: 60%+ of users complete at least one Twelve in Year 1
- **Curation Confidence Score**: 7.5+ average user rating (1-10 scale)

**Secondary Metrics:**

- Photo quality distribution (% of selected photos rated "high quality")
- Temporal spread (photos should span 10+ months of the year)
- Subject diversity (photos should feature multiple family members)
- Time to review (users spend <10 minutes reviewing/adjusting selections)

**How We'll Measure:**

- In-app analytics tracking swap/pin/exclude actions
- Post-review surveys asking "How well did AI capture your year?"
- A/B testing different curation algorithm weights

---

### User Personas

**Primary Users:**

- **Busy Parents (Sarah, 35)** - Wants memories preserved but has no time for manual curation. Trusts AI to "just work."
- **Memory Keeper (Michael, 42)** - Values control; wants to understand curation logic and occasionally override selections

**Secondary Users:**

- **Grandparents (Linda, 68)** - Less tech-savvy; needs automatic curation with minimal interaction
- **Young Professionals (Alex, 28)** - Uses for personal archive ("Me" circle); values aesthetic quality

---

### Dependencies

**Technical Dependencies:**

- Photo storage integration (Google Photos, iCloud, local filesystem)
- Machine learning infrastructure (image classification, face detection)
- Scalable batch processing system (handle millions of photos)

**Product Dependencies:**

- **Requires:** Photo import and storage (foundational infrastructure)
- **Enables:** Epic 2 (Multi-Circle Organization) — curation must work per-circle, not globally
- **Enables:** Epic 3 (Reflection Experience) — curated Twelves are what users revisit

---

### Risks & Mitigations

| Risk                                         | Impact | Probability | Mitigation Strategy                                                   |
| -------------------------------------------- | ------ | ----------- | --------------------------------------------------------------------- |
| AI selections feel random or meaningless     | High   | Medium      | Transparency dashboard showing selection logic; A/B test algorithms   |
| Users don't trust automated curation         | High   | Medium      | Manual override always available; "preview before finalize" flow      |
| Cultural/contextual biases in AI             | High   | Low         | Diverse training data; user feedback loop to identify bias            |
| Performance at scale (millions of photos)    | Medium | Medium      | Batch processing; progressive rollout; caching strategies             |
| Lack of emotional depth in early versions    | Medium | High        | Start with simpler heuristics (quality + diversity); improve over time |

---

### Timeline

**Estimated Duration:** 4-6 months (MVP)

**Phases:**

1. **Phase 1: Foundation (8 weeks)** - Photo quality analyzer, basic selection algorithm
2. **Phase 2: Intelligence (6 weeks)** - Emotional significance detection, diversity engine
3. **Phase 3: Polish (4 weeks)** - Manual override UI, transparency dashboard
4. **Phase 4: Optimization (2-4 weeks)** - Performance tuning, A/B testing

**Key Milestones:**

- **Week 8**: MVP curation engine selecting photos based on quality + temporal diversity
- **Week 14**: Emotional significance integrated; full algorithm operational
- **Week 18**: Manual override interface complete; ready for alpha testing
- **Week 22**: Beta launch with initial user cohort

---

### Open Questions

- [ ] Should curation happen monthly or only at year-end? (Trade-off: engagement vs. finality)
- [ ] How do we handle years with very few photos (<50)? Auto-select all or require minimum?
- [ ] Should users see rejected photos or only selected ones? (Transparency vs. simplicity)
- [ ] Do we need different curation strategies for different circle types? (Family vs. Individual)
- [ ] How do we handle photos with no faces (landscapes, objects)? Weight differently or exclude?

---

### Metadata & Change History

| Version | Date       | Author     | Changes                     |
| ------- | ---------- | ---------- | --------------------------- |
| v1.0    | 2025-10-10 | Chris Park | Initial epic created.       |
