# Tech Spec — TAAC 2026 TokenFormer README

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| react | ^19.0.0 | UI framework |
| react-dom | ^19.0.0 | DOM renderer |
| three | ^0.170.0 | Liquid Lens Refraction Shader |
| @types/three | ^0.170.0 | TypeScript types for Three.js |
| gsap | ^3.12.7 | Scroll-triggered animations + ScrollTrigger + ticker |
| lenis | ^1.2.0 | Smooth scrolling with velocity tracking |
| lucide-react | ^0.468.0 | Icons (Zap, Layers, Trophy, Rocket, ChevronRight, Menu) |
| tailwindcss | ^3.4.19 | Styling |
| typescript | ^5.7.0 | Type safety |
| vite | ^6.0.0 | Build tool |
| @vitejs/plugin-react | ^4.3.0 | React Vite plugin |

**Note**: shadcn/ui Tabs component is already available from init. No additional shadcn packages needed.

---

## Component Inventory

### shadcn/ui Components (Built-in)
| Component | Usage | Installation |
|-----------|-------|--------------|
| Tabs | Architecture section (TokenFormer / Token Stream) | Already installed from init |

### Custom Components

#### Layout
| Component | Props | Description |
|-----------|-------|-------------|
| `Navigation` | — | Fixed glassmorphism nav with 3D text flip links, scroll progress bar |
| `Footer` | — | Minimal footer with centered text |

#### Sections (all rendered in App.tsx)
| Component | Props | Description |
|-----------|-------|-------------|
| `HeroSection` | — | Video bg + Three.js shader overlay + title + badge + CTAs |
| `OverviewSection` | — | Two-column: text left + 2×2 metrics grid right |
| `ArchitectureSection` | — | Tabs with SVG diagram for TokenFormer / Token Stream |
| `CoreResultsSection` | — | Comparison table + AUC bar chart + ASCII timeline |
| `OptimizationJourneySection` | — | Vertical 4-phase timeline with alternating cards |
| `KeyInsightsSection` | — | 2×2 grid of insight cards |
| `FutureDirectionsSection` | — | Horizontal spring carousel with 5 direction cards |
| `ReferencesSection` | — | List of 6 reference papers |

#### Reusable Components
| Component | Props | Description |
|-----------|-------|-------------|
| `SectionHeader` | `overline: string, title: string` | Overline label (Geist Mono uppercase) + oversized title |
| `InsightCard` | `number: string, title: string, description: string, accentColor: string` | Number highlight + title + body with colored left border |
| `MetricCard` | `value: string, label: string, isAccent?: boolean` | Large number + small label in bordered card |
| `FutureCard` | `icon: string, title: string, description: string` | Icon + title + body for carousel |

#### Effects Components (isolated, self-contained)
| Component | Props | Description |
|-----------|-------|-------------|
| `LiquidLensShader` | — | Three.js fullscreen shader overlay (creates own canvas) |
| `HorizontalCarousel` | `children: ReactNode` | Drag-based spring physics carousel wrapper |
| `TextFlipLink` | `href: string, children: string` | 3D rotateX text flip hover effect |
| `ScrollVelocityWrapper` | `children: ReactNode` | Applies skewY/scaleY based on scroll velocity |

#### Hooks
| Hook | Returns | Description |
|------|---------|-------------|
| `useLenis` | `Lenis instance` | Initializes Lenis, connects to GSAP ticker and ScrollTrigger |
| `useScrollReveal` | — | GSAP ScrollTrigger fade-in-up on elements (reusable per section) |
| `useScrollProgress` | `number (0-1)` | Tracks global scroll progress for nav bar |

---

## Animation Implementation Plan

| # | Animation | Library / Approach | Implementation | Complexity |
|---|-----------|-------------------|----------------|------------|
| 1 | **Liquid Lens Refraction Shader** | Three.js raw (manual) | `PlaneGeometry(2,2)` + `OrthographicCamera` + `ShaderMaterial` with custom vertex/fragment shaders. `uTime` from `THREE.Clock`, `uMouse` from window mousemove. Mount/unmount lifecycle in `useEffect`. | **High** 🔒 |
| 2 | **Scroll Velocity Deformation** | Lenis + GSAP | `useLenis` hook tracks `lenis.velocity`. Apply `skewY` and `scaleY` via GSAP to a wrapper div. Cap skewY at ±2deg, scaleY at 1.002. Spring damping toward 0. | **High** 🔒 |
| 3 | **Horizontal Spring Carousel** | Custom (pointer events) | Pointer-down/move/up handlers track drag. On release, compute velocity and animate with requestAnimationFrame spring loop (stiffness: 0.001, damping: 0.0005). Snap to nearest card boundary. | **High** 🔒 |
| 4 | **3D Text Flip Hover** | CSS only | `perspective: 1000px` on container, two text layers stacked. On hover, inner wrapper `rotateX(90deg)` with `transform-style: preserve-3d`. Pure CSS transition `0.5s cubic-bezier(0.76, 0, 0.24, 1)`. | **Medium** |
| 5 | **Hero Entrance Sequence** | GSAP timeline | Master timeline: overline (0.2s delay) → title (0.5s) → subtitle (0.8s) → badge scale+elastic (1.2s) → buttons (1.5s). `gsap.timeline()` with sequential `.from()` calls. | **Medium** |
| 6 | **Scroll-Triggered Section Reveals** | GSAP ScrollTrigger | Reusable pattern: `gsap.from()` with `y: 40, opacity: 0, stagger: 0.1`. Each section calls `useScrollReveal` with its container ref. Trigger at `top 80%`. | **Medium** |
| 7 | **AUC Bar Chart Animation** | GSAP ScrollTrigger | Bars animate `scaleX: 0 → 1` with `transform-origin: left`. Stagger `0.1s`. Triggered on scroll enter. | **Medium** |
| 8 | **Timeline Line Draw** | GSAP ScrollTrigger | Timeline container height animates from `0%` to `100%`. Combined with card reveals alternating from left/right. | **Medium** |
| 9 | **Nav Progress Bar** | CSS + scroll listener | Width percentage bound to `useScrollProgress`. Gradient background. | **Low** |
| 10 | **Nav Link Active State** | Intersection Observer or ScrollTrigger | Track which section is in viewport, update nav link style (color shift + bottom border). | **Low** |
| 11 | **Custom Scrollbar** | CSS only | `::-webkit-scrollbar` selectors with gradient thumb. Firefox `scrollbar-color`. | **Low** |
| 12 | **Card Hover Effects** | CSS only | `transition: border-color 0.3s, box-shadow 0.3s`. Border brightens, subtle glow appears. | **Low** |
| 13 | **Tab Content Fade** | CSS only | `opacity` + `transition: 0.3s ease` on tab panel swap. | **Low** |

---

## State & Logic Plan

### 1. Lenis ↔ GSAP Integration (Global)
Lenis must be initialized once at app root and connected to both GSAP's ticker and ScrollTrigger. The `useLenis` hook owns this singleton initialization. All scroll-dependent features (ScrollTrigger reveals, velocity deformation, progress bar) derive from this single Lenis instance.

**Pattern**: `useLenis` hook returns the Lenis instance. Child components call `lenis.scrollTo()` for nav links. The hook internally wires `lenis.on('scroll', ScrollTrigger.update)` and adds Lenis to `gsap.ticker`.

### 2. Scroll Velocity → CSS Transform (Bridge)
Scroll velocity from Lenis is numeric data that must drive CSS transforms. A `ScrollVelocityWrapper` component reads velocity from the Lenis instance (via `useLenis`) and applies `gsap.set()` to its child wrapper on every Lenis scroll event. The transform is a function of instantaneous velocity with spring damping.

**Key decision**: Apply the deformation to a single wrapper `<div>` rather than individual sections — this is one transform per frame, not N transforms.

### 3. Three.js Shader Lifecycle (Self-contained)
The `LiquidLensShader` component manages its own Three.js scene, camera, renderer, and animation loop entirely within `useEffect`. It creates a `<canvas>` element, appends it to a container ref, and cleans up on unmount. Mouse position is tracked via a window `mousemove` listener (normalized to [0,1]) and fed directly to the `uMouse` uniform. No React state is involved in the render loop — all updates happen via Three.js `requestAnimationFrame`.

### 4. Carousel Drag → Spring Physics (Custom)
The carousel uses pointer events (not mouse events, for mobile compatibility). State machine:
- **Idle**: `translateX` is at a snapped position
- **Dragging**: On `pointermove`, `translateX` follows pointer delta directly
- **Release**: Compute velocity from last 2 frames, run spring animation loop that decays velocity and snaps to nearest card boundary. Use `requestAnimationFrame` — do NOT use React state for the animation loop (use refs for position/velocity to avoid re-renders at 60fps).

### 5. Scroll Progress → Nav Bar (One-way)
`useScrollProgress` computes `(window.scrollY / (document.body.scrollHeight - window.innerHeight))` on Lenis scroll events. Returns a `0-1` number consumed by Navigation for its progress bar width. Simple one-way data flow.

---

## Other Key Decisions

### Raw Three.js (not React Three Fiber)
The shader is a single fullscreen quad with orthographic camera — no scene graph, no 3D objects, no interactivity beyond a mouse uniform. R3F would add unnecessary abstraction. Raw Three.js in a `useEffect` is simpler and more direct for this use case.

### No State Management Library
The app is presentation-only with no shared mutable state beyond the Lenis instance (which is handled by a single hook). All component communication is via props. No Zustand, no Context API needed.

### Font Strategy
Helvetica Now Display and Geist Mono loaded via `@font-face` with `font-display: swap`. Self-host woff2 files in `/public/fonts/`. If fonts fail to load, the system-ui fallback stack maintains visual quality.

### Video Asset
A single `<video>` element in the Hero section. Use `preload="metadata"`, `autoPlay muted loop playsInline`. The video should be a compressed MP4/WebM under 5MB. Place in `/public/videos/hero-bg.mp4`.

### shadcn Tabs Usage
The Architecture section uses the built-in shadcn Tabs component for switching between "TokenFormer 模块" and "统一 Token 流". Tab content is rendered as inline SVG diagrams (no external image dependencies).

### Responsive Breakpoints
- Desktop: > 1024px (full layout)
- Tablet: 768px - 1024px (stacked timeline, 2-card carousel)
- Mobile: < 768px (single column, hamburger nav, swipe carousel)

Tailwind breakpoints: `lg:` (1024px), `md:` (768px).
