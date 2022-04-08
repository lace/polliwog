export type BuiltInValueType = 'boolean' | 'number' | 'string'

// When defining custom types, callers can extend BuiltInValueType:
// export type ValueType = BuiltInValueType | 'MyThing' | 'MyOtherThing'
// export typy MyDependencyGrapy = DependencyGraph<ValueType>

export interface Input<ValueType extends BuiltInValueType> {
  valueType: ValueType
}

export interface ComputeNode<ValueType extends BuiltInValueType> {
  valueType: ValueType
  dependencies: string[]
}

export interface DependencyGraph<ValueType extends BuiltInValueType> {
  schemaVersion: 1
  inputs: { [k: string]: Input<ValueType> }
  intermediates: { [k: string]: ComputeNode<ValueType> }
  outputs: { [k: string]: ComputeNode<ValueType> }
}

export type DependencyGraphWithBuiltInTypes = DependencyGraph<BuiltInValueType>
export type DependencyGraphWithAnyTypes = DependencyGraph<any>
