import * as React from "react";

// Simple implementation to avoid TypeScript errors
export interface Toast {
  id: string;
  title?: React.ReactNode;
  description?: React.ReactNode;
  action?: React.ReactNode;
  open?: boolean;
  variant?: "default" | "destructive";
}

type ToastContextType = {
  toasts: Toast[];
  toast: (props: Omit<Toast, "id">) => string;
  dismiss: (toastId?: string) => void;
  update: (props: Partial<Toast> & { id: string }) => void;
};

const ToastContext = React.createContext<ToastContextType>({
  toasts: [],
  toast: () => "",
  dismiss: () => {}, 
  update: () => {},
});

export function useToast(): ToastContextType {
  return React.useContext(ToastContext);
}
