import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import type { MappingEdge, MappingNode } from '../data/modelGraph'

const palette = {
  verified: '#5be7c4',
  observed: '#58b9ff',
  unknown: '#f1b861',
  blocked: '#fa6d82',
}

type Props = {
  nodes: MappingNode[]
  edges: MappingEdge[]
  selectedId: string
  onSelect: (id: string) => void
}

export function NeuralViewport({ nodes, edges, selectedId, onSelect }: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const selectedRef = useRef(selectedId)
  const onSelectRef = useRef(onSelect)
  selectedRef.current = selectedId
  onSelectRef.current = onSelect

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#07101f')
    scene.fog = new THREE.Fog('#07101f', 12, 28)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.set(0, 1.5, 13)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    mount.appendChild(renderer.domElement)

    const graph = new THREE.Group()
    scene.add(graph)
    const nodeMeshes = new Map<string, THREE.Mesh>()
    const edgeMaterials: Array<{ material: THREE.LineBasicMaterial; base: number }> = []

    const grid = new THREE.GridHelper(18, 24, '#1a4060', '#0d263c')
    grid.position.y = -3.2
    grid.material.transparent = true
    grid.material.opacity = 0.32
    scene.add(grid)
    const ring = new THREE.Mesh(new THREE.TorusGeometry(4.4, 0.012, 8, 96), new THREE.MeshBasicMaterial({ color: '#1c6e91', transparent: true, opacity: 0.36 }))
    ring.rotation.x = Math.PI / 2
    ring.position.y = -2.85
    graph.add(ring)

    scene.add(new THREE.AmbientLight('#9ac6df', 1.2))
    const keyLight = new THREE.PointLight('#55dfff', 14, 18)
    keyLight.position.set(-3, 5, 6)
    scene.add(keyLight)
    const fillLight = new THREE.PointLight('#8d67ff', 10, 16)
    fillLight.position.set(5, -1, 4)
    scene.add(fillLight)

    const positionOf = (id: string) => {
      const node = nodes.find((item) => item.id === id)
      return node ? new THREE.Vector3(...node.position) : new THREE.Vector3()
    }

    edges.forEach((edge) => {
      const start = positionOf(edge.source)
      const end = positionOf(edge.target)
      const geometry = new THREE.BufferGeometry().setFromPoints([start, end])
      const material = new THREE.LineBasicMaterial({ color: edge.relation === 'route' ? '#d8a95c' : '#28617e', transparent: true, opacity: 0.22 + edge.weight * 0.34 })
      const line = new THREE.Line(geometry, material)
      line.userData = { source: edge.source, target: edge.target }
      graph.add(line)
      edgeMaterials.push({ material, base: material.opacity })
    })

    nodes.forEach((node) => {
      const geometry = node.kind === 'model' ? new THREE.IcosahedronGeometry(0.62, 2) : new THREE.SphereGeometry(node.kind === 'layer' ? 0.38 : 0.28, 20, 20)
      const color = palette[node.evidence]
      const material = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.22, roughness: 0.42, metalness: 0.24 })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(...node.position)
      mesh.userData = { nodeId: node.id }
      graph.add(mesh)
      nodeMeshes.set(node.id, mesh)
    })

    const resize = () => {
      const width = mount.clientWidth
      const height = mount.clientHeight
      camera.aspect = width / Math.max(height, 1)
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(mount)

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const orbit = { dragging: false, lastX: 0, lastY: 0, theta: 0, phi: 0.12 }
    const setPointer = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
    }
    const handlePointerDown = (event: PointerEvent) => {
      orbit.dragging = true
      orbit.lastX = event.clientX
      orbit.lastY = event.clientY
      renderer.domElement.setPointerCapture(event.pointerId)
    }
    const handlePointerMove = (event: PointerEvent) => {
      if (!orbit.dragging) return
      orbit.theta += (event.clientX - orbit.lastX) * 0.006
      orbit.phi = Math.max(-0.8, Math.min(0.8, orbit.phi + (event.clientY - orbit.lastY) * 0.004))
      orbit.lastX = event.clientX
      orbit.lastY = event.clientY
    }
    const handlePointerUp = () => { orbit.dragging = false }
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault()
      camera.position.z = Math.max(7, Math.min(18, camera.position.z + event.deltaY * 0.008))
    }
    const handleClick = (event: PointerEvent) => {
      if (Math.abs(event.clientX - orbit.lastX) > 4 || Math.abs(event.clientY - orbit.lastY) > 4) return
      setPointer(event)
      raycaster.setFromCamera(pointer, camera)
      const hit = raycaster.intersectObjects([...nodeMeshes.values()])[0]
      if (hit?.object.userData.nodeId) onSelectRef.current(hit.object.userData.nodeId)
    }
    renderer.domElement.addEventListener('pointerdown', handlePointerDown)
    renderer.domElement.addEventListener('pointermove', handlePointerMove)
    renderer.domElement.addEventListener('pointerup', handlePointerUp)
    renderer.domElement.addEventListener('pointerleave', handlePointerUp)
    renderer.domElement.addEventListener('wheel', handleWheel, { passive: false })
    renderer.domElement.addEventListener('click', handleClick)

    let animationId = 0
    const animate = (time: number) => {
      animationId = requestAnimationFrame(animate)
      graph.rotation.y = orbit.theta + Math.sin(time * 0.00018) * 0.035
      graph.rotation.x = orbit.phi
      ring.rotation.z += 0.0008
      nodeMeshes.forEach((mesh, id) => {
        const active = id === selectedRef.current
        const material = mesh.material as THREE.MeshStandardMaterial
        material.emissiveIntensity = active ? 0.9 : 0.2 + Math.sin(time * 0.0015 + mesh.position.x) * 0.04
        mesh.scale.setScalar(active ? 1.28 : 1)
      })
      edgeMaterials.forEach(({ material, base }, index) => {
        material.opacity = base + Math.sin(time * 0.0012 + index) * 0.025
      })
      renderer.render(scene, camera)
    }
    animationId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(animationId)
      observer.disconnect()
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown)
      renderer.domElement.removeEventListener('pointermove', handlePointerMove)
      renderer.domElement.removeEventListener('pointerup', handlePointerUp)
      renderer.domElement.removeEventListener('pointerleave', handlePointerUp)
      renderer.domElement.removeEventListener('wheel', handleWheel)
      renderer.domElement.removeEventListener('click', handleClick)
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [nodes, edges])

  return <div ref={mountRef} className="h-full min-h-[420px] w-full cursor-grab active:cursor-grabbing" aria-label="Interactive Three.js model mapping graph" />
}
