# Redesign Catan Board Port Visualization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix port visualization on the Catan board dashboard so ports appear on coastal edges with clear indication of the two adjacent settlement vertices that can claim them.

**Architecture:** Replace the current circle-marker approach with edge-aligned port indicators. Each port will be rendered as a visual element positioned ON the edge line between its two vertices, with connection points clearly marked so players understand which two settlement locations claim the port. The visualization will use SVG elements positioned at the edge midpoint with vertex indicators.

**Tech Stack:** 
- JavaScript/SVG for frontend visualization
- Existing dashboard app.js rendering pipeline
- No external libraries

## Global Constraints

- Ports must be clearly distinguishable from other board elements (roads, settlements, dots)
- Port visualization must not obscure vertex/settlement placement
- Ports must render correctly for all board orientations and hex arrangements
- Both 2:1 (resource) and 3:1 (generic) ports must be visually distinct
- Must work with existing board data structure (port.vertices array with two vertex IDs)

---

## File Structure

```
catan_rl/dashboard/static/
  ├── app.js (MODIFY)
  │   └── Port rendering section (lines 217-247)
  │   └── Add helper functions for edge-aligned positioning
  └── style.css (MODIFY)
      └── Add new CSS classes for port visualization (.port-edge, .port-vertex-indicator, etc.)
```

---

## Task 1: Design Port Visualization Style & CSS

**Files:**
- Modify: `catan_rl/dashboard/static/style.css`

**Interfaces:**
- Consumes: Existing color scheme and styling patterns
- Produces: CSS classes for `.port-edge` (main port indicator on edge), `.port-label` (text label), `.vertex-claim-point` (small dot at each vertex)

**Steps:**

- [ ] **Step 1: Add port-edge styling**

In `style.css`, add after the existing port-line/port-marker styles:

```css
.port-edge {
  fill: none;
  stroke: #333;
  stroke-width: 2.5;
  stroke-linecap: round;
}

.port-edge.generic {
  stroke: #666;
}

.port-edge.wheat {
  stroke: #d4af37;
}

.port-edge.sheep {
  stroke: #90ee90;
}

.port-edge.brick {
  stroke: #d2691e;
}

.port-edge.ore {
  stroke: #a9a9a9;
}

.port-edge.wood {
  stroke: #228b22;
}

.vertex-claim-point {
  fill: #f5f5dc;
  stroke: #333;
  stroke-width: 1;
  r: 0.12;
}

.port-label {
  font-size: 0.4em;
  font-weight: bold;
  text-anchor: middle;
  dominant-baseline: central;
  fill: #333;
}
```

- [ ] **Step 2: Verify CSS loads**

Open the browser dashboard at http://localhost:8050 - no visual changes yet (classes not used in HTML).

- [ ] **Step 3: Commit CSS**

```bash
git add catan_rl/dashboard/static/style.css
git commit -m "style: add port edge visualization CSS classes"
```

---

## Task 2: Implement Edge-Aligned Port Positioning Logic

**Files:**
- Modify: `catan_rl/dashboard/static/app.js` (lines 217-247)

**Interfaces:**
- Consumes: `board.ports` array with `port.vertices` (two vertex IDs), `port.resource`
- Produces: SVG line elements positioned on edges with vertex indicator circles

**Steps:**

- [ ] **Step 1: Understand current port rendering**

Read lines 217-247 of app.js to understand:
- How ports are currently positioned (lines 221-225: midpoint + outward offset)
- How the marker is drawn (lines 232-241)
- What data is available (`port.vertices`, `port.resource`)

- [ ] **Step 2: Replace port rendering logic**

Replace the entire port rendering section (lines 218-247) with:

```javascript
  (board.ports || []).forEach((port) => {
    const [va, vb] = port.vertices;
    const pa = vp[va], pb = vp[vb];
    
    // Port edge midpoint (on the actual edge between vertices)
    const mx = (pa[0] + pb[0]) / 2, my = (pa[1] + pb[1]) / 2;
    
    // Determine port type for styling
    const portClass = port.resource === null || port.resource === undefined
      ? "generic"
      : RESOURCE_ABBR[port.resource].toLowerCase();
    
    // Draw thick line ON the edge (replaces old line + offset marker approach)
    const edgeLine = svgEl("line", {
      x1: tx(pa[0]), y1: ty(pa[1]),
      x2: tx(pb[0]), y2: ty(pb[1]),
      class: `port-edge ${portClass}`,
    });
    staticLayer.appendChild(edgeLine);
    
    // Draw small circles at each vertex to show settlement claim points
    [pa, pb].forEach((vertex) => {
      const claimDot = svgEl("circle", {
        cx: tx(vertex[0]), cy: ty(vertex[1]),
        class: "vertex-claim-point",
      });
      staticLayer.appendChild(claimDot);
    });
    
    // Draw port label at edge midpoint
    const text = svgEl("text", {
      x: tx(mx), y: ty(my),
      class: "port-label",
    });
    text.textContent = port.resource === null || port.resource === undefined
      ? "3:1"
      : RESOURCE_ABBR[port.resource];
    
    // Add tooltip showing full resource name
    const fullLabel = port.resource === null || port.resource === undefined
      ? "3:1 generic"
      : `${RESOURCE_NAMES[port.resource]} 2:1`;
    const title = svgEl("title", {});
    title.textContent = fullLabel;
    text.appendChild(title);
    
    staticLayer.appendChild(text);
  });
```

- [ ] **Step 3: Test port visualization in browser**

1. Ensure dashboard is running: `http://localhost:8050`
2. Click on a game with traces
3. Verify ports now appear:
   - As colored lines ON the coastal edges (not offset outward)
   - With small circles at both end vertices (settlement claim points)
   - With label text centered on the edge
   - Colors match resource types (different color for generic 3:1)

- [ ] **Step 4: Verify all port types render correctly**

Check dashboard shows:
- ✅ Generic 3:1 ports with gray edge lines
- ✅ Resource 2:1 ports with resource-colored edges
- ✅ Small claim-point dots at both vertices of each port edge
- ✅ Clear labels (3:1 or resource abbreviation)

- [ ] **Step 5: Commit visualization fix**

```bash
git add catan_rl/dashboard/static/app.js
git commit -m "fix(dashboard): render ports on edges with vertex claim indicators"
```

---

## Task 3: Verify Port Rendering With Multiple Board Orientations

**Files:**
- Test: View dashboard across different game traces
- Verify: `catan_rl/dashboard/static/app.js` rendering (unchanged after Task 2)

**Steps:**

- [ ] **Step 1: Test with different game traces**

Open dashboard and click through different games (iter0000_game0000 through game0008) to verify ports render correctly across different board states.

- [ ] **Step 2: Verify no overlaps with settlements/roads**

Check that:
- Port edge lines don't visually overlap with road lines
- Vertex claim points are visible even if settlement exists there
- Port labels don't obscure game state

- [ ] **Step 3: Check edge cases**

Verify corner cases:
- Ports on board edges (should still render as edge lines)
- Multiple ports on same vertex area (should each have their own edge line)
- Port rendering when settlements/roads are present

- [ ] **Step 4: Take screenshot for documentation**

Capture a board view showing:
- Correct port edge alignment
- Vertex claim points at both settlement locations
- Multiple port types with different colors
- Save as: `docs/port-visualization-fixed.png` (optional, for reference)

- [ ] **Step 5: Document fix in dashboard README**

If a dashboard README exists, add note:
```
## Port Visualization

Ports are rendered as colored lines on coastal edges:
- Each port connects TWO adjacent vertices where settlements can claim it
- Small circles at each vertex show where settlements can be placed
- Resource ports use resource colors; 3:1 generic ports use gray
- Port labels (resource abbreviation or "3:1") appear at edge center
```

---

## Verification Checklist

Before considering this complete:

- [ ] All ports render on edges, not in hex centers
- [ ] Each port shows TWO vertex claim points (small circles)
- [ ] Port colors are consistent and distinguishable
- [ ] Labels are readable and positioned correctly
- [ ] Ports don't overlap with roads or obscure settlements
- [ ] Works across all game traces in dashboard
- [ ] No console errors when rendering ports
- [ ] Git history shows clean commits with clear messages

---

## Visual Verification Examples

After completing Task 2, verify you see:

**Generic 3:1 Port:**
- Gray line along a coastal edge
- Two small circles at edge endpoints
- "3:1" label at edge center

**Resource 2:1 Port:**
- Colored line (e.g., yellow/gold for wheat, green for sheep)
- Two small circles at edge endpoints
- Resource abbreviation (e.g., "W" for wheat) at edge center

**Result:** Players can clearly see that a port provides access from TWO adjacent settlement locations, matching official Catan rules.
