# Design Vision — Remember Twelve

## Design Philosophy

**"Timeless Simplicity for Lasting Memories"**

Remember Twelve's design must feel like a cherished photo album, not a tech product. Every interface choice should prioritize **emotional resonance over technical complexity**, **clarity over cleverness**, and **durability over trends**. We're designing for moments families will revisit for decades—the aesthetic must age as gracefully as the memories it preserves.

### Core Philosophy Tenets

1. **Heirloom Quality**: Design that feels permanent, not disposable
2. **Invisible Technology**: AI works behind the scenes; users experience magic, not mechanics
3. **Emotional First**: Every interaction should feel warm, personal, never transactional
4. **Progressive Simplicity**: Simple by default; complexity available when needed

---

## Visual Style

### Aesthetic Direction

**Inspiration**: Physical photo albums, museum exhibits, handcrafted journals

**Mood**: Warm, nostalgic, intimate, timeless

**Not**: Corporate, sterile, gamified, trendy

### Color Palette

**Primary:**
- **Warm Neutrals**: Creams, soft grays, subtle taupes (backgrounds, containers)
- **Accent**: Soft amber/gold (highlights, year markers, special moments)

**Secondary:**
- **Deep Charcoal**: Text, headers (high contrast, readability)
- **Muted Blues/Greens**: Subtle accents for circles, categories

**Avoid**: Bright primaries, neon, high-saturation colors (feel dated quickly)

### Typography

**Primary Font**: Serif for headlines (warmth, tradition)
- Examples: Freight Text, Tiempos, Lyon

**Secondary Font**: Sans-serif for body text (clarity, readability)
- Examples: Inter, SF Pro, Helvetica Neue

**Hierarchy**:
- Large, generous spacing (breathable, elegant)
- Minimal font weights (Regular, Medium, Bold only)

### Photography Treatment

- **Full-Bleed Images**: Photos are the hero; no heavy borders or frames
- **Grid Layouts**: Clean, organized, museum-quality presentation
- **Minimal Overlays**: Text on photos only when essential; always readable
- **Aspect Ratio Flexibility**: Respect original photo proportions; no forced cropping

### Iconography

- **Minimal**: Icons only where universally understood (share, export, settings)
- **Outlined Style**: Matches timeless, understated aesthetic
- **Never Decorative**: Every icon serves a function

---

## Design Principles

### 1. **Clarity Over Cleverness**

Every interaction should be immediately understandable. No "magic gestures" or hidden features that require tutorials.

**Examples**:
- Timeline scrolls naturally (vertical, not carousel)
- "Swap Photo" button appears on hover—obvious, not buried
- Circle names are user-editable text, not icons

---

### 2. **Focus on Content, Not Chrome**

Photos are the product. UI should recede, not compete for attention.

**Examples**:
- Minimal navigation bars (hide when scrolling)
- No decorative backgrounds or patterns
- White space is generous; let photos breathe

---

### 3. **Emotional Transitions**

Every transition should feel meaningful, not mechanical. Animation serves storytelling.

**Examples**:
- Year-over-year comparisons dissolve smoothly
- Timeline scrubbing feels fluid, responsive
- Photo swaps animate naturally (slide out/in, not pop)

---

### 4. **Inclusive by Default**

Design for all ages, abilities, and tech literacy levels.

**Examples**:
- Large touch targets (minimum 44x44pt)
- High contrast text (WCAG AAA compliance)
- Simple language (no jargon, no "syncing your vectors")

---

### 5. **Platform-Native Feel**

Respect OS conventions while maintaining brand identity.

**Examples**:
- iOS: Native gestures (swipe back), SF Symbols where appropriate
- Android: Material patterns (FABs, bottom sheets)
- Web: Keyboard shortcuts, familiar patterns

---

## User Experience Goals

### What We're Creating

**For Busy Parents**: An app that "just works"—open once a year, see your Twelve, share with grandparents, done.

**For Memory Keepers**: A place to browse, reflect, and occasionally tweak—satisfying control without overwhelming complexity.

**For Grandparents**: An experience as simple as flipping through a photo album—intuitive, comforting, never intimidating.

### Experience Pillars

1. **Effortless Discovery**: Users find past Twelves easily; browsing feels natural
2. **Delightful Surprise**: AI curation reveals forgotten moments; "I forgot about this!"
3. **Confident Control**: Manual overrides are obvious; users never feel trapped by AI
4. **Ritualized Reflection**: Year-end flow becomes an anticipated tradition

---

## Accessibility

### Commitment

Remember Twelve will meet **WCAG 2.1 Level AA** standards minimum, aiming for **AAA** where feasible.

### Key Considerations

**Vision**:
- Minimum 4.5:1 contrast ratio for body text (7:1 for AAA)
- Scalable text (support Dynamic Type on iOS, font scaling on Android/Web)
- Never rely on color alone for meaning (use icons, labels, patterns)

**Motor**:
- Large touch targets (44x44pt minimum)
- Generous spacing between interactive elements
- No time-based interactions (e.g., hold-to-activate)

**Cognitive**:
- Simple, consistent navigation
- Clear labels (no cryptic icons)
- Undo always available (especially for photo swaps)

**Screen Readers**:
- Semantic HTML (web)
- Proper accessibility labels (iOS/Android)
- Alt text for all meaningful images

---

## Responsiveness

### Device Strategy

Remember Twelve is **mobile-first** but **multi-platform** by design.

**Priority Platforms**:
1. **iOS Mobile** (iPhone) — Primary platform, 60% of target users
2. **Android Mobile** — Secondary, 30% of users
3. **Web (Desktop)** — Tertiary, 10% of users (primarily for setup, export)

### Responsive Patterns

**Mobile (Portrait)**:
- Single-column layouts
- Full-screen photo views
- Bottom navigation
- Swipe gestures for navigation

**Tablet (Landscape)**:
- Two-column layouts (sidebar + content)
- Split-screen year comparisons
- Persistent navigation

**Desktop (Web)**:
- Three-column layouts where appropriate
- Keyboard shortcuts (arrow keys for timeline, esc to close)
- Hover states for secondary actions

### Breakpoints

- **Small**: <375px (iPhone SE)
- **Medium**: 375-768px (Most phones)
- **Large**: 768-1024px (Tablets)
- **XLarge**: >1024px (Desktop)

---

## Interaction Patterns

### Navigation

**Primary**: Bottom tab bar (mobile) or sidebar (tablet/desktop)
- Home / Circles / Timeline / Settings

**Secondary**: Contextual (in-screen headers, back buttons)

### Gestures (Mobile)

- **Swipe Left/Right**: Navigate between Twelves in timeline
- **Pinch**: Zoom into photos
- **Long Press**: Reveal photo actions (swap, share, details)
- **Pull to Refresh**: Update photo library sync

### Feedback

- **Haptics**: Confirm actions (photo swapped, Twelve saved)
- **Toasts**: Non-intrusive success messages
- **Errors**: Inline, clear, actionable (never just "Error 404")

---

## Design System

### Component Library

Build a modular, reusable component library:

**Core Components**:
- Photo Grid (responsive, lazy-loaded)
- Timeline Scrubber (horizontal year selector)
- Circle Card (visual representation of each circle)
- Year-over-Year Comparison View
- Photo Swap Modal
- Empty States (onboarding, no photos)

**Tools**:
- Figma for design
- Storybook for component documentation (web)
- SwiftUI/Jetpack Compose for native components

---

## Final Note

This design vision will evolve as we learn from users—but the **core philosophy remains fixed**: timeless, simple, emotionally resonant. Every design decision should pass this test:

**"Would a family still want to open this app in 2045?"**

If the answer is no, redesign.
