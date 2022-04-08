export type Vector3 = [number, number, number]

export interface Polyline {
  vertices: Vector3[]
  isClosed: boolean
}

export interface Plane {
  referencePoint: Vector3
  unitNormal: Vector3
}
