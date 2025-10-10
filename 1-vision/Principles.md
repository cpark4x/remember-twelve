# Product Principles — Remember Twelve

## Core Principles

These principles guide every decision we make—from feature prioritization to design choices to technical architecture.

### 1. **Effortless by Default, Control When Needed**

Remember Twelve must work automatically with zero configuration—but provide granular control for those who want it.

- **Default**: AI curates twelve photos per year automatically
- **Control**: Users can swap photos, adjust circles, or override selections
- **Why**: Most users want to set-and-forget; power users want agency

**Decision Test**: If a feature requires configuration before it delivers value, it fails this principle.

---

### 2. **Preservation Over Perfection**

Better to preserve imperfect memories automatically than to wait for perfect curation that never happens.

- **Embrace "Good Enough"**: An automatically curated Twelve delivered on time beats a manually perfected one created never
- **Iterate Over Time**: Curation algorithms improve; users can refine past Twelves as the system learns
- **Why**: The enemy of memory preservation is perfectionism

**Decision Test**: Would this feature delay or complicate automatic preservation? If yes, reconsider.

---

### 3. **Multi-Circle by Design**

Every feature must work across multiple circles (Family, Individual, Kids, Cousins, etc.), not just one archive.

- **No Single Timeline**: Avoid assuming one user = one archive
- **Circle-Aware**: All features (sharing, printing, viewing) must handle multiple circles gracefully
- **Why**: Families are complex; memory preservation should reflect that reality

**Decision Test**: Does this feature assume a single archive? If yes, redesign.

---

### 4. **Timeless Design, Future-Proof Data**

Visual design and data formats must age gracefully across decades.

- **Classic Over Trendy**: Avoid design fads; embrace timeless aesthetics
- **Open Formats**: Store data in formats that will remain accessible in 30 years
- **Platform Independence**: Never lock memories into proprietary systems
- **Why**: These memories must outlast us, our devices, and potentially the company itself

**Decision Test**: Will this look dated or be inaccessible in 2050? If maybe, reconsider.

---

### 5. **AI as Curator, Not Creator**

AI selects and organizes existing photos—it never fabricates or manipulates memories.

- **No Generative Changes**: AI doesn't alter photo content, only selects and arranges
- **Transparent Curation**: Users can always see why a photo was chosen
- **Human Override**: Final say always belongs to the user
- **Why**: Trust is paramount; memories must remain authentic

**Decision Test**: Does this feature generate or alter photo content? If yes, don't build it.

---

### 6. **Reflection is the Product**

The value isn't in storage—it's in revisiting. Optimize for the moment families look back.

- **Browsing > Searching**: Enable serendipitous rediscovery, not just keyword search
- **Yearly Ritual**: Build features that encourage annual reflection
- **Compare Across Time**: Show how families change year over year
- **Why**: Preserved memories have no value unless they're revisited

**Decision Test**: Does this feature encourage reflection or just more storage? Prioritize the former.

---

### 7. **Start Simple, Scale Complexity**

Begin with the simplest possible implementation; add complexity only when essential.

- **V1 is Opinionated**: Fewer choices, more automation
- **Progressive Disclosure**: Advanced features appear only when needed
- **No Feature Bloat**: Each addition must justify its complexity cost
- **Why**: Simplicity compounds; complexity multiplies

**Decision Test**: Can we ship without this feature? If yes, defer it.

---

## Decision Framework

When faced with trade-offs, use this hierarchy:

1. **Preservation First**: Does this ensure memories are captured and stored?
2. **Ease Over Power**: Does this make the default experience simpler?
3. **Multi-Circle Support**: Does this work for all circles, not just one?
4. **Timeless Design**: Will this age well over decades?
5. **Reflection Value**: Does this encourage revisiting memories?

If principles conflict, **Preservation First** wins. Better to have imperfect memories than no memories.

---

## Non-Goals

What we are **explicitly NOT** trying to do:

### **Not a Social Network**
- We're not competing with Instagram or Facebook
- Sharing exists, but it's private and family-focused
- No public profiles, follower counts, or engagement metrics

### **Not a Photo Editor**
- We curate, not edit
- Basic cropping/rotation acceptable; filters and effects are not our focus
- Users can edit elsewhere and re-import

### **Not a General-Purpose Photo Storage**
- We're not replacing Google Photos or iCloud
- We're a curated layer on top of existing storage
- Users keep their full libraries elsewhere

### **Not a Print-on-Demand Service (Initially)**
- Printing partnerships may come later
- V1 focuses on digital preservation and curation
- Users can export and print anywhere

### **Not an AI Photo Generator**
- We never create or alter photo content with AI
- AI is for selection and organization only
- Authenticity over enhancement

### **Not a Comprehensive Family Tree**
- We organize by "circles," not genealogical relationships
- If family tree features emerge, they're lightweight and user-defined
- We're not competing with Ancestry or MyHeritage

---

## Guiding Questions

When evaluating new features or resolving conflicts, ask:

- **Does this make preservation more automatic or more manual?** (Choose automatic)
- **Does this add complexity to the default experience?** (If yes, reconsider)
- **Will this matter in 20 years?** (If no, deprioritize)
- **Does this serve reflection or storage?** (Prioritize reflection)
- **Would we be proud to show this to a family with young children?** (Our core user)

These principles are living guidelines. They evolve as we learn—but they should change slowly and deliberately, not reactively.
