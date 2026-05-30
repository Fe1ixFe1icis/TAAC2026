import { useEffect, useRef } from 'react';
import * as THREE from 'three';

const vertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = `
uniform float uTime;
uniform vec2 uMouse;
uniform float uDisplacementScale;
uniform vec2 uResolution;
varying vec2 vUv;

vec4 refractionMap(vec2 uv, float scale) {
  vec2 displacement = vec2(
    sin(uv.y * 10.0 + uTime * 0.5 + uMouse.x * 2.0) * 0.1,
    cos(uv.x * 10.0 + uTime * 0.3 + uMouse.y * 2.0) * 0.1
  ) * scale;
  
  // Create flowing neural network-like pattern
  float pattern = sin(uv.x * 8.0 + displacement.x * 5.0 + uTime * 0.4) 
                * cos(uv.y * 6.0 + displacement.y * 5.0 + uTime * 0.3);
  pattern += sin(uv.x * 12.0 - uTime * 0.2) * cos(uv.y * 10.0 + uTime * 0.5) * 0.5;
  
  // Color based on pattern - deep blue/purple palette
  vec3 color1 = vec3(0.04, 0.02, 0.05); // near black
  vec3 color2 = vec3(0.15, 0.1, 0.25);  // deep purple
  vec3 color3 = vec3(0.1, 0.12, 0.35);  // deep blue
  vec3 color4 = vec3(0.05, 0.2, 0.3);   // deep teal
  
  float t = pattern * 0.5 + 0.5;
  vec3 color = mix(color1, color2, t);
  color = mix(color, color3, smoothstep(0.3, 0.7, t));
  color = mix(color, color4, smoothstep(0.6, 1.0, t) * 0.5);
  
  // Add subtle glow near mouse
  float mouseDist = length(uv - uMouse);
  float glow = exp(-mouseDist * mouseDist * 8.0) * 0.15;
  color += vec3(0.2, 0.15, 0.4) * glow;
  
  // Vignette
  float vignette = 1.0 - smoothstep(0.3, 1.2, length(uv - 0.5) * 1.5);
  color *= vignette * 0.7 + 0.3;
  
  return vec4(color, 1.0);
}

void main() {
  vec4 color = refractionMap(vUv, uDisplacementScale);
  gl_FragColor = color;
}
`;

export default function LiquidLensShader() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mouseRef = useRef({ x: 0.5, y: 0.5 });
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const renderer = new THREE.WebGLRenderer({ alpha: false, antialias: false });
    
    renderer.setSize(container.offsetWidth, container.offsetHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    container.appendChild(renderer.domElement);

    // Uniforms
    const uniforms = {
      uTime: { value: 0.0 },
      uMouse: { value: new THREE.Vector2(0.5, 0.5) },
      uDisplacementScale: { value: 2.0 },
      uResolution: { value: new THREE.Vector2(container.offsetWidth, container.offsetHeight) },
    };

    // Mesh
    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms,
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    // Clock
    const clock = new THREE.Clock();

    // Mouse handler
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = e.clientX / window.innerWidth;
      mouseRef.current.y = 1.0 - e.clientY / window.innerHeight;
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Animation loop
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      uniforms.uTime.value = clock.getElapsedTime();
      uniforms.uMouse.value.x += (mouseRef.current.x - uniforms.uMouse.value.x) * 0.05;
      uniforms.uMouse.value.y += (mouseRef.current.y - uniforms.uMouse.value.y) * 0.05;
      renderer.render(scene, camera);
    };
    animate();

    // Resize handler
    const handleResize = () => {
      const w = container.offsetWidth;
      const h = container.offsetHeight;
      renderer.setSize(w, h);
      uniforms.uResolution.value.set(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 1,
        pointerEvents: 'none',
      }}
    />
  );
}
