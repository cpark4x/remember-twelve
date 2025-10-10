# Epic 2: Multi-Circle Memory Organization

---

### Problem Statement

Family memories don't fit into a single timeline—they're multi-dimensional. A photo from a family vacation belongs in "Family," "Kids," and "Extended Family" circles simultaneously. Current photo tools force users into single-album thinking, requiring duplicate organization or abandoning nuanced memory categorization. Users need a system that recognizes memory complexity and automatically organizes photos into multiple meaningful circles without manual duplication.

---

### Objective

Build a multi-circle organization system that allows users to create distinct memory archives (Family, Individual, Kids, Cousins, etc.), with photos intelligently assigned to relevant circles and each circle generating its own annual Twelve—all while maintaining a unified photo library and avoiding duplication.

---

### Business Value

- **Core Differentiator**: Multi-circle architecture is Remember Twelve's unique value proposition vs. competitors
- **Increased Engagement**: Users with multiple circles return more frequently (2.5x sessions vs. single-circle users)
- **Network Effects**: Shared circles (e.g., "Extended Family") create viral loops as users invite family members
- **Long-term Retention**: More circles = more preserved memories = stronger lock-in

---

### Scope

**In-Scope:**

- Circle creation and management (create, rename, archive)
- Automatic photo assignment to circles based on people, context, and metadata
- Manual photo assignment and reassignment
- Per-circle curation (each circle gets its own Twelve)
- Shared circles with family members (invite-based)
- Circle privacy controls (private vs. shared)

**Out-of-Scope:**

- Advanced genealogy/family tree features (not competing with Ancestry)
- Public or social circles (all circles remain private or invite-only)
- Circle-based messaging or social features (we're not a social network)
- Automated circle suggestions beyond basic defaults (Family, Me, Kids)

---

### Features

This epic comprises the following features:

1. **Feature 2.1: Circle Creation & Management** - Create, name, customize, and archive circles
2. **Feature 2.2: Intelligent Photo Assignment** - Automatically assign photos to circles based on faces, context, dates
3. **Feature 2.3: Manual Photo Organization** - Drag-and-drop interface for photo assignment across circles
4. **Feature 2.4: Shared Circle Invitations** - Invite family members to contribute to shared circles
5. **Feature 2.5: Per-Circle Curation** - Each circle runs independent AI curation for its Twelve
6. **Feature 2.6: Circle Archive View** - Timeline view showing all Twelves for a specific circle

---

### Success Metrics

**Primary Metrics:**

- **Multi-Circle Adoption**: 40% of users have 2+ active circles (Year 1 target)
- **Circle Engagement**: 70%+ of created circles have active photos and views
- **Shared Circle Acceptance**: 60%+ of shared circle invitations accepted

**Secondary Metrics:**

- Average circles per user (target: 2.5)
- Photos per circle (indicates proper organization)
- Cross-circle photo overlap (how many photos appear in multiple circles)
- Circle abandonment rate (<20%)

**How We'll Measure:**

- Track circle creation, photo additions, and view activity
- Monitor invitation send/accept rates
- Survey users on circle usage patterns and satisfaction

---

### User Personas

**Primary Users:**

- **Busy Parents (Sarah, 35)** - Wants separate "Family" and "Kids" circles; doesn't want work event photos in family archive
- **Memory Keeper (Michael, 42)** - Creates circles for "Extended Family," "High School Friends," "Travel"; values organization
- **Grandparents (Linda, 68)** - Part of shared "Extended Family" circle; contributes photos from grandkids' visits

**Secondary Users:**

- **Young Professionals (Alex, 28)** - Uses "Me" circle for personal archive; "Friends" circle for social memories
- **Extended Family (Uncle Tom, 55)** - Receives invite to shared circle; contributes holiday photos

---

### Dependencies

**Technical Dependencies:**

- User authentication and permissions system (for shared circles)
- Face recognition and photo metadata analysis (for automatic assignment)
- Real-time sync infrastructure (for shared circle updates)

**Product Dependencies:**

- **Requires:** Epic 1 (AI Curation) must support per-circle curation, not just global
- **Requires:** Photo import and storage infrastructure
- **Enables:** Future social features (if we build them) rely on circle architecture

---

### Risks & Mitigations

| Risk                                             | Impact | Probability | Mitigation Strategy                                                        |
| ------------------------------------------------ | ------ | ----------- | -------------------------------------------------------------------------- |
| Users don't understand circle concept            | High   | Medium      | Clear onboarding with defaults (Family, Me, Kids); progressive disclosure  |
| Automatic photo assignment feels inaccurate      | High   | Medium      | Manual override always available; learn from corrections                   |
| Shared circles create family conflict           | Medium | Low         | Clear ownership/permission model; ability to leave or hide circles         |
| Too many circles leads to fragmentation          | Medium | Medium      | Limit circle count initially (5-7 max); guide users toward simplicity      |
| Cross-circle photo management becomes complex    | Medium | High        | Simple multi-select UI; "assign to all circles" default for shared moments |

---

### Timeline

**Estimated Duration:** 5-7 months (MVP)

**Phases:**

1. **Phase 1: Foundation (6 weeks)** - Circle data model, basic create/manage UI
2. **Phase 2: Intelligence (6 weeks)** - Automatic photo assignment based on faces and metadata
3. **Phase 3: Collaboration (8 weeks)** - Shared circles, invitation system, permissions
4. **Phase 4: Polish (4 weeks)** - Timeline views, multi-select UX, onboarding flow
5. **Phase 5: Integration (2-4 weeks)** - Per-circle curation integration with Epic 1

**Key Milestones:**

- **Week 6**: Circle CRUD (create, read, update, delete) operational
- **Week 12**: Automatic photo assignment working for default circles
- **Week 20**: Shared circles with invite system complete
- **Week 24**: Per-circle Twelve generation integrated; ready for beta

---

### Open Questions

- [ ] Should we limit the number of circles users can create? (Prevent fragmentation vs. flexibility)
- [ ] How do we handle photos that don't clearly belong to any circle? (Unassigned pool or force assignment?)
- [ ] Should users be able to merge or split circles later? (Flexibility vs. archive stability)
- [ ] Do shared circles allow all members to curate the Twelve, or only the owner? (Democracy vs. chaos)
- [ ] How do we handle deleted shared circles? (Archive for all members or only owner?)

---

### Metadata & Change History

| Version | Date       | Author     | Changes                     |
| ------- | ---------- | ---------- | --------------------------- |
| v1.0    | 2025-10-10 | Chris Park | Initial epic created.       |
