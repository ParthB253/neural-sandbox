import { createContext } from 'react';
export type NodeAction = 'add' | 'edit' | 'transforms' | 'connect' | 'delete';
export const EditorContext = createContext<{ act: (action: NodeAction, id: string) => void; disabled: boolean }>({ act: () => {}, disabled: false });
