# Aesthetic Guide: Remember Twelve

**Created:** October 28, 2025
**Status:** Active v1.0

---

## User's Vision (Preserved)

**Raw input:**
"I want to design this tool so it looks modern, clean, and highly polished. Something that feels like a chic Seoul coffee shop or like Maru Coffee in LA."

**Refined priorities (in order):**
1. Modern
2. Natural
3. Clean
4. Warm
5. Sophisticated

**Reference inspiration:**
- Maru Coffee LA aesthetic
- Chic Seoul coffee shop atmosphere
- Contemporary understated luxury

---

## Emotional Direction

### Primary Feeling: Refined Warmth

**Personality adjectives:** Modern, natural, clean, warm, sophisticated, timeless, approachable, premium

### What Users Should Feel

**First impression:** "This looks beautifully crafted"
**During use:** "This feels calm and intentional"
**After use:** "I trust this with my memories"

### The Coffee Shop Connection

Like walking into Maru Coffee, Remember Twelve should feel:
- **Welcoming but refined** - You're invited in, not intimidated
- **Spacious and breathing** - Generous room to think and reflect
- **Tactile and real** - Digital that feels human, not synthetic
- **Contemporary classic** - Current but won't feel dated in 10 years

---

## Visual Language Framework

### The Maru Coffee Translation

From coffee shop to productivity tool:

**Coffee Shop Quality** → **Digital Expression**
- Warm wood tones → Warm neutral backgrounds (#F3F3F3, soft creams)
- Natural light flooding space → Ample whitespace, generous breathing room
- Muted teal/sage accents → Sophisticated accent colors that don't shout
- Handcrafted touches → Subtle texture, refined details
- Minimalist furniture → Clean components, purposeful elements
- Soft ambient music → Smooth, deliberate motion timing

---

## Color Philosophy

### Foundation: Warm Neutrals

**Primary Background Palette:**
```css
--surface-primary: #FAFAF8;    /* Warm white - not pure white */
--surface-secondary: #F3F3F3;  /* Light warm gray - Maru inspired */
--surface-tertiary: #EEEDE9;   /* Soft taupe - tactile depth */
--surface-elevated: #FFFFFF;   /* Pure white for cards/elevation */
```

**Rationale:** These aren't cold grays (blue-tinted). They're warm, natural neutrals that feel like paper, linen, unbleached cotton. They provide calm without sterility.

### Accents: Sophisticated Restraint

**Muted Natural Tones:**
```css
--accent-sage: #7396A2;        /* Muted teal/sage - Maru inspired */
--accent-sage-deep: #5487A0;   /* Deeper sage - interactive states */
--accent-amber: #C9A869;       /* Soft amber/gold - special moments */
--accent-terracotta: #C89B7F;  /* Warm clay - secondary highlights */
```

**Rationale:** These are desaturated, sophisticated tones. Not bright or primary. They feel natural (sage from plants, amber from light, terracotta from earth) and work harmoniously without competing.

### Text: Warm Charcoal

**Text Hierarchy:**
```css
--text-primary: #2A2825;       /* Warm charcoal - not pure black */
--text-secondary: #5A5551;     /* Mid-tone warm gray */
--text-tertiary: #8A857F;      /* Light warm gray - subtle text */
--text-inverse: #FAFAF8;       /* For dark backgrounds */
```

**Rationale:** Text with warm undertones maintains the coffee shop feeling. Pure black (#000000) would feel too harsh and digital. This is readable but human.

### Usage Philosophy

**When to use each color:**

**Warm neutrals (backgrounds):**
- Primary surface: Main app background
- Secondary surface: Containers, panels
- Tertiary surface: Subtle differentiation within cards
- Elevated surface: Floating cards, modals, popovers

**Sage accent:**
- Circle indicators (Family, Kids, Cousins)
- Selected states in navigation
- Secondary buttons and actions
- Subtle borders and dividers

**Amber accent:**
- Year markers in timeline (special significance)
- Primary call-to-action buttons (sparingly)
- Highlights for featured photos or moments
- Success states and confirmations

**Terracotta accent:**
- Alternative circle colors for variety
- Warm interactive states (hover, focus)
- Occasional highlight for secondary emphasis

**Key principle:** Restraint. Let photos be the color. UI fades to warm neutrals with occasional sophisticated accents.

---

## Shadow & Depth Strategy

### Philosophy: Subtle Elevation, Not Drama

Maru Coffee has understated depth - nothing heavy or harsh. Shadows suggest physical layers without demanding attention.

### Four-Layer Shadow System

For elevated components (cards, modals, floating buttons):

```css
--shadow-soft:
  0 0 0 1px rgba(42, 40, 37, 0.04),           /* Border definition */
  0 1px 2px -1px rgba(42, 40, 37, 0.06),      /* Near shadow */
  0 2px 4px -1px rgba(42, 40, 37, 0.04);      /* Ambient depth */

--shadow-medium:
  0 0 0 1px rgba(42, 40, 37, 0.04),
  0 2px 4px -1px rgba(42, 40, 37, 0.06),
  0 4px 8px -2px rgba(42, 40, 37, 0.08),
  0 8px 16px -4px rgba(42, 40, 37, 0.06);     /* Floating depth */

--shadow-strong:
  0 0 0 1px rgba(42, 40, 37, 0.04),
  0 4px 8px -2px rgba(42, 40, 37, 0.08),
  0 8px 16px -4px rgba(42, 40, 37, 0.10),
  0 16px 32px -8px rgba(42, 40, 37, 0.12);    /* Modal/overlay depth */
```

### When to Use Shadows

**Soft shadows:**
- Photo cards at rest
- Container separation
- Subtle component boundaries

**Medium shadows:**
- Interactive cards on hover
- Navigation bars
- Toolbars and action bars

**Strong shadows:**
- Modals and overlays
- Photo detail views
- Floating action buttons

**No shadows:**
- Flat content areas
- Text-only sections
- Timeline scrubbers
- Most buttons (rely on background color instead)

### Alternative: Border-Based Depth

For flatter areas that still need definition:

```css
--border-subtle: 1px solid rgba(42, 40, 37, 0.06);
--border-medium: 1px solid rgba(42, 40, 37, 0.10);
```

Use borders when:
- Component is on same elevation plane
- Visual distinction needed without floating effect
- Maintaining maximum flatness

---

## Corner Treatment

### Philosophy: Soft but Measured

Maru Coffee uses gentle 5px corners - not sharp (cold/harsh) but not overly rounded (playful/casual).

### Corner Radius Scale

```css
--radius-small: 6px;      /* Small elements: tags, badges, chips */
--radius-medium: 8px;     /* Standard: buttons, inputs, cards */
--radius-large: 12px;     /* Larger surfaces: modals, panels */
--radius-xlarge: 16px;    /* Hero elements: photo grids, feature cards */
--radius-circle: 50%;     /* Profile avatars, icon buttons */
```

### Usage Guidelines

**Small (6px):**
- Tags and labels
- Small interactive chips
- Inline badges

**Medium (8px) - PRIMARY:**
- Standard buttons
- Text inputs and form fields
- Photo cards in grids
- Navigation items
- Most UI components

**Large (12px):**
- Modal containers
- Large card surfaces
- Panel headers
- Feature sections

**XLarge (16px):**
- Hero photo displays
- Year summary cards
- Major feature cards

**Rationale:** 8px is the baseline - soft enough to feel approachable, refined enough to feel premium. Not 4px (too subtle), not 16px (too playful for a tool handling precious memories).

---

## Typography Voice

### Philosophy: Warm Serif + Clean Sans

Following the Design Vision's established direction, enhanced with coffee shop sophistication.

### Type Pairing

**Headlines/Emotional Text:** Serif (warmth, tradition, timelessness)
**Body/Interface Text:** Sans-serif (clarity, modernity, readability)

### Recommended Type System

**Primary (Headlines):**
- Freight Text Pro
- Tiempos Headline
- Lyon Display
- (Fallback: Georgia, serif)

**Secondary (Body/UI):**
- Inter (recommended - excellent for UI)
- SF Pro (native iOS feel)
- Helvetica Neue
- (Fallback: system-ui, sans-serif)

### Type Scale

```css
/* Display - Hero moments */
--text-display: 48px / 1.1 / 600;     /* Year titles, hero headlines */

/* Headline - Section headers */
--text-h1: 32px / 1.2 / 600;          /* Main page titles */
--text-h2: 24px / 1.3 / 600;          /* Section headers */
--text-h3: 20px / 1.4 / 600;          /* Subsection headers */

/* Body - Reading text */
--text-body-large: 18px / 1.5 / 400;  /* Comfortable reading */
--text-body: 16px / 1.5 / 400;        /* Standard body text */
--text-body-small: 14px / 1.5 / 400;  /* Secondary information */

/* UI - Interface elements */
--text-ui: 15px / 1.4 / 500;          /* Buttons, navigation */
--text-ui-small: 13px / 1.4 / 500;    /* Labels, captions */
--text-ui-tiny: 11px / 1.3 / 600;     /* Metadata, timestamps */
```

Format: `size / line-height / weight`

### Font Weight Philosophy

**Use only 3-4 weights maximum:**
- **Regular (400):** Body text, descriptions
- **Medium (500):** UI elements, navigation, emphasis
- **Semibold (600):** Headlines, important labels
- (Optional) **Bold (700):** Rare, only for critical emphasis

**Avoid:** Thin (300), ExtraBold (800), Black (900) - they feel too extreme for this aesthetic.

### Line Height: Generous Breathing

**Reading content:** 1.5-1.6× (comfortable, spacious)
**Headlines:** 1.1-1.3× (tighter, more impact)
**UI elements:** 1.4× (balanced between compact and readable)

### Letter Spacing: Subtle Refinement

```css
--tracking-tight: -0.02em;    /* Display headlines only */
--tracking-normal: 0;         /* Body text, most UI */
--tracking-loose: 0.02em;     /* Small caps, tiny labels */
```

---

## Spacing & Layout Philosophy

### The Coffee Shop Principle: Room to Breathe

Maru Coffee feels spacious even when busy. Generous whitespace is not wasteful - it's respectful.

### Spacing Scale (8px base unit)

```css
--space-1: 4px;      /* Tight: icon-text gaps */
--space-2: 8px;      /* Small: element padding */
--space-3: 12px;     /* Medium: component gaps */
--space-4: 16px;     /* Standard: card padding */
--space-5: 24px;     /* Large: section spacing */
--space-6: 32px;     /* XLarge: major sections */
--space-7: 48px;     /* XXLarge: page sections */
--space-8: 64px;     /* Generous: hero spacing */
```

### Layout Principles

**Whitespace budget:** 40-50% of screen should be breathing room

**Content width:**
- Reading text: max 680px (optimal line length ~60-75 characters)
- Photo grids: responsive, but generous gaps
- Modals: max 600px (comfortable focus)

**Vertical rhythm:**
- Maintain consistent spacing between sections (use --space-5/6/7)
- Avoid arbitrary values (always use spacing scale)
- More space before headers than after (visual grouping)

**Grid philosophy:**
- 12-column grid for flexibility
- Generous gutters: 24px mobile, 32px desktop
- Align to grid but prioritize optical balance

---

## Texture & Materiality

### Philosophy: Digital that Feels Tactile

Coffee shops have natural materials - wood grain, paper menus, ceramic mugs. Digital can evoke this without skeuomorphism.

### Subtle Texture Techniques

**1. Noise overlay (ultra-subtle):**
```css
background:
  url('data:image/svg+xml,...') /* 1-2% opacity grain texture */,
  linear-gradient(...);
```
- Adds organic quality to flat surfaces
- Barely visible but subconsciously warm
- Use on large surface areas only

**2. Soft gradients (linear, subtle):**
```css
background: linear-gradient(
  180deg,
  #FAFAF8 0%,
  #F3F3F3 100%
);
```
- Mimics natural light fall-off
- Adds dimensionality without shadows
- Keep gradient shifts under 5-8% difference

**3. Blur/translucency (depth perception):**
```css
backdrop-filter: blur(12px) saturate(120%);
background: rgba(250, 250, 248, 0.92);
```
- For overlays, modals, navigation bars
- Creates depth through layering
- Maintains hierarchy without heaviness

### When to Use Texture

**Do:**
- Large background surfaces (subtle grain)
- Elevated UI elements (soft gradients)
- Overlays and modals (blur effects)

**Don't:**
- Small components (noise would be muddy)
- Photo areas (interferes with content)
- Text containers (reduces readability)

---

## Motion Personality

### Philosophy: Deliberate and Smooth

Coffee shop movements are unhurried. A barista doesn't slam things down - they place them carefully. Motion should feel intentional, not instant or bouncy.

### Timing Scale

```css
--duration-instant: 100ms;    /* Micro-interactions: hover states */
--duration-fast: 200ms;       /* Quick: button press, toggle */
--duration-medium: 350ms;     /* Standard: modal open, navigation */
--duration-slow: 500ms;       /* Deliberate: page transitions */
--duration-extra-slow: 700ms; /* Ceremonial: year reveal */
```

### Easing Philosophy

**Primary easing (use 80% of the time):**
```css
--ease-standard: cubic-bezier(0.4, 0.0, 0.2, 1);  /* Material standard */
```
- Smooth deceleration
- Natural feeling
- Not too bouncy, not too linear

**Entrance easing:**
```css
--ease-enter: cubic-bezier(0.0, 0.0, 0.2, 1);     /* Accelerate */
```
- Elements entering view
- Quick start, smooth finish

**Exit easing:**
```css
--ease-exit: cubic-bezier(0.4, 0.0, 1, 1);        /* Decelerate */
```
- Elements leaving view
- Smooth start, quick finish

**Gentle spring (sparingly):**
```css
--ease-spring: cubic-bezier(0.34, 1.15, 0.64, 1); /* Subtle overshoot */
```
- Special interactions only
- Photo swaps, celebratory moments
- Never on large movements (would feel unnatural)

### Motion Principles

**1. Stagger reveals:**
When showing multiple items (photo grid loading):
```css
transition-delay: calc(var(--index) * 50ms);
```
- Progressive disclosure
- Guides eye naturally
- Feels crafted, not instant

**2. Respect reduced motion:**
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```
- Always honor user preferences
- Accessibility first

**3. Purpose over decoration:**
- Motion should communicate state changes
- Avoid animation for animation's sake
- Every movement has intent

---

## Interaction Patterns

### Touch Targets

**Minimum sizes:**
- Mobile: 44×44pt (iOS standard)
- Desktop: 32×32pt minimum, 40×40pt preferred

**Spacing between targets:**
- Mobile: 8-12px minimum
- Desktop: 4-8px minimum

### Button States

**Primary Button (Amber accent - use sparingly):**
```css
/* Rest */
background: var(--accent-amber);
color: var(--text-primary);

/* Hover */
background: #D4B578; /* Lighter amber */
transform: translateY(-1px);
box-shadow: var(--shadow-medium);

/* Active */
background: #B8965A; /* Darker amber */
transform: translateY(0);

/* Focus (keyboard) */
outline: 2px solid var(--accent-sage);
outline-offset: 2px;
```

**Secondary Button (Sage accent):**
```css
/* Rest */
background: transparent;
border: 1.5px solid var(--accent-sage);
color: var(--accent-sage-deep);

/* Hover */
background: rgba(115, 150, 162, 0.08);
border-color: var(--accent-sage-deep);
```

**Ghost Button (Minimal):**
```css
/* Rest */
background: transparent;
color: var(--text-secondary);

/* Hover */
background: rgba(42, 40, 37, 0.04);
color: var(--text-primary);
```

### Photo Interactions

**Photo Card:**
```css
/* Rest */
border-radius: var(--radius-medium);
box-shadow: var(--shadow-soft);
transition: all 350ms var(--ease-standard);

/* Hover */
transform: scale(1.02);
box-shadow: var(--shadow-medium);

/* Selected */
outline: 3px solid var(--accent-sage);
outline-offset: 3px;
```

**Photo Grid:**
- Consistent gaps: 16px mobile, 24px desktop
- Lazy load with fade-in: 300ms opacity transition
- Aspect ratio preservation (no forced crops)

---

## Component-Specific Aesthetics

### Cards

**Standard Card:**
```css
background: var(--surface-elevated);
border-radius: var(--radius-medium);
padding: var(--space-4);
box-shadow: var(--shadow-soft);
```

**Photo Card:**
```css
background: var(--surface-elevated);
border-radius: var(--radius-medium);
overflow: hidden; /* Clip photo to rounded corners */
box-shadow: var(--shadow-soft);
```

**Circle Card (for organizing groups):**
```css
background: linear-gradient(135deg,
  var(--surface-secondary) 0%,
  var(--surface-tertiary) 100%
);
border-radius: var(--radius-large);
padding: var(--space-5);
border: 1px solid var(--accent-sage);
```

### Navigation

**Bottom Tab Bar (mobile):**
```css
background: rgba(250, 250, 248, 0.95);
backdrop-filter: blur(16px) saturate(120%);
border-top: 1px solid rgba(42, 40, 37, 0.06);
padding: var(--space-2) var(--space-4);
```

**Sidebar (desktop):**
```css
background: var(--surface-secondary);
border-right: 1px solid var(--border-subtle);
padding: var(--space-5);
width: 240px;
```

### Modals

**Modal Container:**
```css
background: var(--surface-elevated);
border-radius: var(--radius-large);
box-shadow: var(--shadow-strong);
padding: var(--space-6);
max-width: 600px;
```

**Modal Backdrop:**
```css
background: rgba(42, 40, 37, 0.60);
backdrop-filter: blur(8px);
```

### Forms

**Text Input:**
```css
background: var(--surface-primary);
border: 1.5px solid var(--border-medium);
border-radius: var(--radius-medium);
padding: var(--space-3) var(--space-4);
font-size: var(--text-body);
color: var(--text-primary);
transition: all 200ms var(--ease-standard);

/* Focus */
border-color: var(--accent-sage);
box-shadow: 0 0 0 3px rgba(115, 150, 162, 0.12);
outline: none;
```

---

## Design Tokens Reference

### Complete Token System

```css
/* ============================================
   REMEMBER TWELVE - DESIGN TOKENS
   Aesthetic: Modern Natural Refined
   Inspiration: Maru Coffee LA
   ============================================ */

/* -------------------- Colors -------------------- */

/* Surfaces */
--surface-primary: #FAFAF8;
--surface-secondary: #F3F3F3;
--surface-tertiary: #EEEDE9;
--surface-elevated: #FFFFFF;

/* Accents */
--accent-sage: #7396A2;
--accent-sage-deep: #5487A0;
--accent-amber: #C9A869;
--accent-terracotta: #C89B7F;

/* Text */
--text-primary: #2A2825;
--text-secondary: #5A5551;
--text-tertiary: #8A857F;
--text-inverse: #FAFAF8;

/* Borders */
--border-subtle: 1px solid rgba(42, 40, 37, 0.06);
--border-medium: 1px solid rgba(42, 40, 37, 0.10);
--border-strong: 1px solid rgba(42, 40, 37, 0.16);

/* -------------------- Shadows -------------------- */

--shadow-soft:
  0 0 0 1px rgba(42, 40, 37, 0.04),
  0 1px 2px -1px rgba(42, 40, 37, 0.06),
  0 2px 4px -1px rgba(42, 40, 37, 0.04);

--shadow-medium:
  0 0 0 1px rgba(42, 40, 37, 0.04),
  0 2px 4px -1px rgba(42, 40, 37, 0.06),
  0 4px 8px -2px rgba(42, 40, 37, 0.08),
  0 8px 16px -4px rgba(42, 40, 37, 0.06);

--shadow-strong:
  0 0 0 1px rgba(42, 40, 37, 0.04),
  0 4px 8px -2px rgba(42, 40, 37, 0.08),
  0 8px 16px -4px rgba(42, 40, 37, 0.10),
  0 16px 32px -8px rgba(42, 40, 37, 0.12);

/* -------------------- Corner Radius -------------------- */

--radius-small: 6px;
--radius-medium: 8px;
--radius-large: 12px;
--radius-xlarge: 16px;
--radius-circle: 50%;

/* -------------------- Spacing -------------------- */

--space-1: 4px;
--space-2: 8px;
--space-3: 12px;
--space-4: 16px;
--space-5: 24px;
--space-6: 32px;
--space-7: 48px;
--space-8: 64px;

/* -------------------- Typography -------------------- */

/* Fonts */
--font-serif: 'Freight Text Pro', 'Tiempos', Georgia, serif;
--font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro', sans-serif;

/* Sizes (with line-height) */
--text-display: 48px / 1.1;
--text-h1: 32px / 1.2;
--text-h2: 24px / 1.3;
--text-h3: 20px / 1.4;
--text-body-large: 18px / 1.5;
--text-body: 16px / 1.5;
--text-body-small: 14px / 1.5;
--text-ui: 15px / 1.4;
--text-ui-small: 13px / 1.4;
--text-ui-tiny: 11px / 1.3;

/* Weights */
--weight-regular: 400;
--weight-medium: 500;
--weight-semibold: 600;

/* Tracking */
--tracking-tight: -0.02em;
--tracking-normal: 0;
--tracking-loose: 0.02em;

/* -------------------- Motion -------------------- */

/* Duration */
--duration-instant: 100ms;
--duration-fast: 200ms;
--duration-medium: 350ms;
--duration-slow: 500ms;
--duration-extra-slow: 700ms;

/* Easing */
--ease-standard: cubic-bezier(0.4, 0.0, 0.2, 1);
--ease-enter: cubic-bezier(0.0, 0.0, 0.2, 1);
--ease-exit: cubic-bezier(0.4, 0.0, 1, 1);
--ease-spring: cubic-bezier(0.34, 1.15, 0.64, 1);

/* -------------------- Layout -------------------- */

--max-width-text: 680px;
--max-width-modal: 600px;
--gutter-mobile: 24px;
--gutter-desktop: 32px;
```

---

## Usage Guidelines

### Do's

- Use warm neutrals for all backgrounds and surfaces
- Apply soft shadows (8px medium radius) to elevated components
- Use sage accent for navigation, circles, secondary actions
- Use amber accent sparingly for primary CTAs and special moments
- Maintain generous whitespace (40-50% of screen)
- Honor the 8px spacing scale consistently
- Prioritize photo content over UI chrome
- Use serif fonts for emotional headers, sans for UI
- Implement smooth, deliberate motion (350ms standard)
- Let photos provide the color palette

### Don'ts

- Don't use pure white (#FFF) as primary background (too harsh)
- Don't use pure black (#000) for text (too digital)
- Don't use bright saturated colors (breaks sophisticated feel)
- Don't use sharp corners (0px = too cold) or overly round (24px+ = too playful)
- Don't use instant transitions (<100ms) or very slow (>700ms)
- Don't use heavy drop shadows or glows (feels dated)
- Don't use decorative textures that interfere with photos
- Don't use more than 3-4 font weights
- Don't create dense layouts with insufficient breathing room
- Don't add motion just for decoration

---

## Cross-Reference: Existing Design Vision Alignment

This aesthetic guide builds upon and refines the Design Vision established in `/Users/chrispark/amplifier/remember-twelve/3-design/DesignVision.md`.

### Key Alignments

**Design Vision says:** "Warm, nostalgic, intimate, timeless"
**Aesthetic Guide provides:** Warm neutrals (#FAFAF8), muted natural accents (sage/amber), soft corners (8px), generous spacing

**Design Vision says:** "Heirloom Quality - Design that feels permanent, not disposable"
**Aesthetic Guide provides:** Timeless color palette, restrained motion, sophisticated typography, lasting material quality

**Design Vision says:** "Focus on Content, Not Chrome"
**Aesthetic Guide provides:** 40-50% whitespace budget, minimal UI decoration, photos as hero with receding interface

**Design Vision says:** "Emotional Transitions - Animation serves storytelling"
**Aesthetic Guide provides:** Deliberate timing (350ms standard), smooth easing (cubic-bezier), staggered reveals, purposeful motion

**Design Vision says:** "Clarity Over Cleverness"
**Aesthetic Guide provides:** Clean visual hierarchy, readable contrast (4.5:1+), intuitive interactive states, accessible touch targets

### What This Guide Adds

The Design Vision provided **philosophy and direction**.
This Aesthetic Guide provides **systematic implementation** - specific color values, shadow formulas, timing curves, spacing scales, and usage rules that designers and developers can reference to build components that feel "Maru Coffee modern-natural-sophisticated."

---

## Reference Examples

### Example 1: Photo Card Component

**Visual treatment:**
- Background: `--surface-elevated` (#FFFFFF)
- Border radius: `--radius-medium` (8px)
- Shadow: `--shadow-soft` (subtle, multi-layer)
- Padding: `--space-4` (16px)
- Hover state: Scale 1.02, shadow upgrade to `--shadow-medium`, 350ms transition

**Why it works:**
- Soft corners feel approachable but refined
- Subtle shadow creates gentle elevation
- Hover feedback is smooth, not jarring
- White card background makes photo pop against warm neutral app background

### Example 2: Year Timeline Scrubber

**Visual treatment:**
- Background: `--surface-secondary` with subtle gradient to `--surface-tertiary`
- Selected year: Amber accent (`--accent-amber`) with medium weight serif font
- Unselected years: `--text-tertiary` with sans-serif
- Transition: 350ms with `--ease-standard` as you scrub
- Spacing: Years separated by `--space-5` (24px)

**Why it works:**
- Warm gradient background feels tactile, like aged paper
- Amber highlight signals "this year matters" emotionally
- Smooth transitions make scrubbing feel fluid, not jerky
- Generous spacing prevents crowding, feels breathable

### Example 3: Circle Selection Card

**Visual treatment:**
- Background: Gradient from `--surface-secondary` to `--surface-tertiary`
- Border: 1px solid `--accent-sage` (subtle frame)
- Border radius: `--radius-large` (12px - slightly more rounded for friendliness)
- Label: Serif font, `--text-h3` size
- Icon: Outlined style, `--accent-sage`
- Padding: `--space-5` (24px)

**Why it works:**
- Gradient adds dimensionality without heavy shadows
- Sage border ties to navigation/circle system
- Larger radius (12px) appropriate for larger card surface
- Serif label adds warmth and personality to functional card
- Generous padding makes card feel substantial, not cramped

### Example 4: Primary CTA Button

**Visual treatment:**
- Background: `--accent-amber` (#C9A869)
- Text: `--text-primary` (warm charcoal - high contrast)
- Border radius: `--radius-medium` (8px)
- Padding: `--space-3` vertical, `--space-4` horizontal
- Font: Sans-serif, `--weight-medium` (500)
- Hover: Lighten amber 10%, translate Y -1px, add `--shadow-medium`, 200ms fast transition
- Active: Darken amber 10%, translate Y 0, 100ms instant

**Why it works:**
- Amber is used sparingly - only for primary actions
- High contrast text ensures readability
- Soft corners match system-wide treatment
- Hover lift + shadow feels responsive and tactile
- Fast transitions on button press feel immediate without being jarring

---

## Evolution & Maintenance

### Last Updated
**October 28, 2025** - Initial creation based on user vision: "Modern, natural, clean, warm, sophisticated - like Maru Coffee LA"

### When to Update This Guide

Update when:
- User provides aesthetic feedback ("feels too warm", "needs more contrast", etc.)
- New features require new component patterns
- Usability testing reveals visual clarity issues
- Platform conventions change (iOS/Android design updates)
- Brand identity evolves

### Update Process

1. **Document the feedback** - Add to "User's Vision" section
2. **Adjust principles** - Update relevant sections (color, shadow, motion, etc.)
3. **Maintain rationale** - Explain why changes were made
4. **Version control** - Increment version number, add to evolution log
5. **Communicate changes** - Notify design-system-architect and component-designer

### Evolution Log

**v1.0 (Oct 28, 2025)** - Initial aesthetic guide created
- Captured "Maru Coffee modern-natural-warm-sophisticated" vision
- Established warm neutral color palette with muted sage/amber accents
- Defined soft corner system (8px primary), subtle shadow strategy, deliberate motion timing
- Set typography pairing (serif headlines + sans UI)
- Documented generous spacing philosophy (40-50% whitespace)

---

## Final Note: The Coffee Shop Test

Every design decision should pass this test:

**"Would this feel at home in Maru Coffee?"**

If it's too bright, too harsh, too busy, too cold, too trendy - it doesn't belong.

This aesthetic is about:
- **Restraint over abundance**
- **Warmth over sterility**
- **Timelessness over trendiness**
- **Breathing room over density**
- **Natural materials over synthetic**
- **Sophistication over decoration**

When in doubt, choose the warmer neutral. Choose the softer corner. Choose the gentler shadow. Choose the more deliberate motion. Choose space over clutter.

This is a tool for preserving precious memories. It should feel as carefully crafted and lovingly maintained as the photos it holds.

---

**End of Aesthetic Guide v1.0**
